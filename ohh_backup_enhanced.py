import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import os
import imageio
import io
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)


class TradingEnv(gym.Env):
    """
    The final, refined environment to teach the agent to
    "cut losses short and let winners run."
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window=40):
        super(TradingEnv, self).__init__()
        self.df = df.dropna().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_fee_percent = 0.0005 # 0.05% per transaction

        self.num_features = 7 
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback_window, self.num_features), dtype=np.float32
        )
        self.reset()

    def _get_obs(self):
        start = max(self.current_step - self.lookback_window + 1, 0)
        end = self.current_step + 1
        frame = self.df.iloc[start:end].copy()

        in_position_flag = 1.0 if self.position_open else 0.0
        unrealized_profit = 0.0
        if self.position_open:
            current_price = self.df['close'].iloc[self.current_step]
            unrealized_profit = (current_price - self.entry_price) / self.entry_price
        
        frame['in_position'] = in_position_flag
        frame['unrealized_profit'] = unrealized_profit
        
        if len(frame) < self.lookback_window:
            padding = pd.DataFrame(np.zeros((self.lookback_window - len(frame), len(frame.columns))), columns=frame.columns)
            frame = pd.concat([padding, frame], ignore_index=True)
        
        return frame[['price_vs_mavg', 'bandwidth', 'rsi', 'macd', 'stoch_k', 'in_position', 'unrealized_profit']].values.astype(np.float32)

    def _get_info(self):
        return { 'net_worth': self.net_worth, 'balance': self.balance, 'position_open': self.position_open,
                 'entry_price': self.entry_price, 'total_trades': len(self.trades) }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window - 1 
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position_open = False
        self.entry_price = 0
        self.entry_step = 0
        self.trades = []
        self.last_balance = self.initial_balance
        self.history = []
        return self._get_obs(), self._get_info()

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        if self.position_open:
            trade_duration = self.current_step - self.entry_step
            unrealized_profit_pct = (current_price - self.entry_price) / self.entry_price
            if unrealized_profit_pct > 0:
                reward += 0.01 
            else:
                reward -= 0.01 * (trade_duration / 10)
                
        if action == 1 and not self.position_open:
            self.position_open = True
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.last_balance = self.balance
            fee = self.balance * self.transaction_fee_percent
            self.balance -= fee
        
        elif action == 2 and self.position_open:
            exit_price = current_price
            position_value = self.last_balance * (exit_price / self.entry_price)
            sell_fee = position_value * self.transaction_fee_percent
            self.balance = position_value - sell_fee
            
            net_profit_percent = (self.balance - self.last_balance) / self.last_balance * 100 if self.last_balance > 0 else 0
            
            reward += net_profit_percent 
            
            if net_profit_percent > 3.0:
                reward += 10
            
            trade = {'entry_step': self.entry_step, 'exit_step': self.current_step,
                     'entry_price': self.entry_price, 'exit_price': exit_price, 'profit_%': net_profit_percent}
            self.trades.append(trade)
            self.position_open = False
        
        if self.position_open: self.net_worth = self.last_balance * (current_price / self.entry_price)
        else: self.net_worth = self.balance
        
        self.history.append({
            'step': self.current_step, 
            'net_worth': self.net_worth, 
            'action': action, 
            'price': current_price,
            'balance': self.balance,
            'position_open': 1.0 if self.position_open else 0.0
        })

        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            done = False
            self.current_step += 1
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, False, info


def preprocess_data(filepath):
    """Loads data, resamples to 1H, and correctly normalizes."""
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('date').set_index('date')
    
    aggregation = { 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum' }
    df = df.resample('1h').agg(aggregation).dropna()
    
    # Explicitly calculate and concat indicators
    bb_df = df.ta.bbands(length=20)
    rsi_ser = df.ta.rsi(length=14)
    macd_df = df.ta.macd(fast=12, slow=26)
    stoch_df = df.ta.stoch(k=14, d=3)
    
    df = pd.concat([df, bb_df, rsi_ser, macd_df, stoch_df], axis=1)
    
    rename_dict = {}
    for col in df.columns:
        if 'BBL' in col: rename_dict[col] = 'lower_band'
        if 'BBM' in col: rename_dict[col] = 'middle_band'
        if 'BBU' in col: rename_dict[col] = 'upper_band'
        if 'RSI' in col: rename_dict[col] = 'rsi'
        if 'MACD_12_26_9' == col: rename_dict[col] = 'macd'
        if 'STOCHk_14_3_3' == col: rename_dict[col] = 'stoch_k'
        
    df = df.rename(columns=rename_dict)
    
    epsilon = 1e-10
    df['price_vs_mavg'] = (df['close'] - df['middle_band']) / (df['upper_band'] - df['lower_band'] + epsilon)
    df['bandwidth'] = (df['upper_band'] - df['lower_band']) / (df['middle_band'] + epsilon)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    cols_to_normalize = ['rsi', 'macd', 'stoch_k']
    for col in cols_to_normalize:
        mean = train_df[col].mean()
        std = train_df[col].std()
        train_df[col] = (train_df[col] - mean) / (std + epsilon)
        test_df[col] = (test_df[col] - mean) / (std + epsilon)
    
    return train_df, test_df


def run_detailed_backtest(env, model):
    """
    Runs a backtest and prints a comprehensive, forensic performance report.
    """
    obs, info = env.reset()
    num_steps = len(env.df) - env.lookback_window
    for _ in tqdm(range(num_steps), desc="Backtesting"):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated: break

    if env.position_open:
        exit_price = env.df['close'].iloc[env.current_step]
        position_value = env.last_balance * (exit_price / env.entry_price)
        sell_fee = position_value * env.transaction_fee_percent
        env.balance = position_value - sell_fee
        env.net_worth = env.balance
        net_profit = env.balance - env.last_balance
        net_profit_percent = (net_profit / env.last_balance) * 100 if env.last_balance > 0 else 0
        trade = {'entry_step': env.entry_step, 'exit_step': env.current_step,
                 'entry_price': env.entry_price, 'exit_price': exit_price, 'profit_%': net_profit_percent}
        env.trades.append(trade)

    print("\n" + "="*60)
    print(" " * 15 + "FORENSIC BACKTESTING REPORT")
    print("="*60)

    initial_balance = env.initial_balance
    final_balance = env.net_worth
    trades = env.trades
    if not trades:
        print("\nNo trades were made during the backtest.")
        print("="*60)
        return pd.DataFrame(), pd.DataFrame(env.history)

    trades_df = pd.DataFrame(trades)
    if 'entry_step' not in trades_df.columns or 'exit_step' not in trades_df.columns or trades_df.empty:
        print("\nTrade data is incomplete or missing.")
        return trades_df, pd.DataFrame(env.history)
        
    # We'll set the dates later in the analytics section using the correct test_df index
    
    print("\n--- I. Core Performance Metrics ---")
    total_net_profit = final_balance - initial_balance
    total_net_profit_pct = (total_net_profit / initial_balance) * 100
    print(f"Total Net Profit:        ${total_net_profit:,.2f} ({total_net_profit_pct:.2f}%)")

    history_df = pd.DataFrame(env.history)
    history_df['returns'] = history_df['net_worth'].pct_change()
    trading_periods_per_year = 24 * 365
    sharpe_ratio = (history_df['returns'].mean() / history_df['returns'].std()) * np.sqrt(trading_periods_per_year) if history_df['returns'].std() > 0 else 0
    history_df['cumulative_max'] = history_df['net_worth'].cummax()
    history_df['drawdown_pct'] = ((history_df['net_worth'] - history_df['cumulative_max']) / history_df['cumulative_max']) * 100
    max_drawdown = history_df['drawdown_pct'].min()
    print(f"Max Drawdown:            {max_drawdown:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    print("\n--- II. Trade-Level Performance ---")
    profits = trades_df['profit_%']
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    print(f"Total Trades:            {len(trades_df)}")
    print(f"Win Rate:                {(len(wins) / len(trades_df) * 100 if not trades_df.empty else 0):.2f}%")
    if not wins.empty and not losses.empty:
        print(f"Profit Factor:           {(wins.sum() / abs(losses.sum())):.2f}")
    
    print(f"Best Trade:              {profits.max():.2f}%")
    print(f"Worst Trade:             {profits.min():.2f}%")
    print(f"Average Trade P/L:       {profits.mean():.2f}%")
    if wins.empty: print(f"Average Win P/L:         N/A")
    else: print(f"Average Win P/L:         {wins.mean():.2f}%")
    if losses.empty: print(f"Average Loss P/L:        N/A")
    else: print(f"Average Loss P/L:        {losses.mean():.2f}%")

    print("\n--- III. Streaks & Consistency ---")
    trades_df['win'] = trades_df['profit_%'] > 0
    consecutive_wins, max_consecutive_wins, consecutive_losses, max_consecutive_losses = 0, 0, 0, 0
    for is_win in trades_df['win']:
        if is_win:
            consecutive_wins += 1; consecutive_losses = 0
        else:
            consecutive_losses += 1; consecutive_wins = 0
        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    print(f"Max Consecutive Wins:      {max_consecutive_wins}")
    print(f"Max Consecutive Losses:    {max_consecutive_losses}")
        
    print("\n" + "="*60)
    return trades_df, history_df


def calculate_advanced_metrics(history_df, trades_df, initial_balance):
    """Computes PhD-level financial metrics."""
    if history_df.empty: return {}
    
    max_dd = history_df['drawdown_pct'].min() / 100
    
    # Sharpe & Sortino (Annualized 24*365 for hourly)
    ann_factor = 24 * 365
    mean_ret = history_df['returns'].mean()
    std_ret = history_df['returns'].std()
    sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0
    
    downside_ret = history_df[history_df['returns'] < 0]['returns']
    sortino = (mean_ret / downside_ret.std() * np.sqrt(ann_factor)) if not downside_ret.empty and downside_ret.std() > 0 else 0
    
    # Calmar (CAGR / MaxDD)
    total_days = (len(history_df) / 24)
    total_return = (history_df['net_worth'].iloc[-1] / initial_balance) - 1
    cagr = (1 + total_return)**(365 / total_days) - 1 if total_days > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    # Benchmark (Buy & Hold)
    benchmark_return = (history_df['price'].iloc[-1] / history_df['price'].iloc[0]) - 1
    
    # Rolling Sharpe (30 day window = 720 hours)
    history_df['rolling_sharpe'] = history_df['returns'].rolling(window=720).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(ann_factor)) if x.std() > 0 else 0
    )
    
    return {
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'cagr': cagr
    }


def create_professional_dashboard(df, history_df, trades_df, metrics):
    """Generates a high-impact Plotly dashboard."""
    # Ensure DatetimeIndex for resampling
    history_df = history_df.copy()
    history_df.index = df.index[history_df['step']]
    
    if not isinstance(history_df.index, pd.DatetimeIndex):
        history_df.index = pd.to_datetime(history_df.index)

    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['entry_date'] = df.index[trades_df['entry_step']]
        trades_df['exit_date'] = df.index[trades_df['exit_step']]
        trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Equity Curve: Agent vs Benchmark", "Drawdown Profile",
            "Daily Returns Distribution", "Rolling Sharpe Ratio (30D)",
            "Win Rate by Trade Duration", "Leverage Impact Simulation (Theoretical)",
            "Trade P/L Distribution", "Cumulative Returns comparison"
        ),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Equity Curve
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['net_worth'], name="Agent Equity", line=dict(color='cyan', width=2)), row=1, col=1)
    # Benchmark scaling
    bench_price = history_df['price'] / history_df['price'].iloc[0] * history_df['net_worth'].iloc[0]
    fig.add_trace(go.Scatter(x=history_df.index, y=bench_price, name="Benchmark (BTC)", line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # 2. Drawdown
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['drawdown_pct'], name="Drawdown %", fill='tozeroy', line=dict(color='red')), row=1, col=2)
    
    # 3. Daily Returns Distribution
    daily_rets = history_df['returns'].resample('D').sum()
    fig.add_trace(go.Histogram(x=daily_rets, name="Daily Returns", nbinsx=50, marker_color='rgb(158,202,225)', opacity=0.75), row=2, col=1)
    
    # 4. Rolling Sharpe
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['rolling_sharpe'], name="Rolling Sharpe", line=dict(color='orange')), row=2, col=2)
    
    # 5. Win Rate by Duration
    if not trades_df.empty:
        trades_df['is_win'] = trades_df['profit_%'] > 0
        trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
        wr_by_dur = trades_df.groupby(pd.cut(trades_df['duration_hours'], bins=[0, 5, 12, 24, 48, 100, 500]), observed=True)['is_win'].mean() * 100
        fig.add_trace(go.Bar(x=[str(b) for b in wr_by_dur.index], y=wr_by_dur.values, name="Win Rate %", marker_color='lightgreen'), row=3, col=1)
    
    # 6. Leverage Impact
    lev_fs = [1, 2, 5]
    for l in lev_fs:
        lev_pnl = (history_df['returns'] * l).add(1).cumprod() * history_df['net_worth'].iloc[0]
        fig.add_trace(go.Scatter(x=history_df.index, y=lev_pnl, name=f"{l}x Leverage"), row=3, col=2)
        
    # 7. Trade P/L Distribution
    if not trades_df.empty:
        fig.add_trace(go.Box(x=trades_df['profit_%'], name="Trade P/L %", marker_color='plum'), row=4, col=1)
        
    # 8. Cumulative Returns Comparison
    fig.add_trace(go.Scatter(x=history_df.index, y=(history_df['net_worth']/history_df['net_worth'].iloc[0]-1)*100, name="Agent Ret %"), row=4, col=2)
    fig.add_trace(go.Scatter(x=history_df.index, y=(history_df['price']/history_df['price'].iloc[0]-1)*100, name="BTC Ret %"), row=4, col=2)

    fig.update_layout(height=1400, width=1200, title_text="Advanced Performance Analytics Dashboard", template="plotly_dark", showlegend=True)
    
    dashboard_file_html = "performance_dashboard.html"
    dashboard_file_png = "portfolio_summary_hd.png"
    
    fig.write_html(dashboard_file_html)
    print(f"\nProfessional dashboard saved to {dashboard_file_html}")
    
    # Save as HD PNG for GitHub
    try:
        fig.write_image(dashboard_file_png, scale=2, width=1200, height=1400)
        print(f"HD Dashboard image saved to {dashboard_file_png}")
    except Exception as e:
        print(f"Could not save HD image: {e}")

    try:
        fig.show()
    except:
        pass


def generate_trading_gif(df, history_df, n_steps=100, filename="trading_live.gif"):
    """Generates an animated GIF of the agent trading live."""
    print(f"Generating animated trading GIF (last {n_steps} steps)...")
    
    frames = []
    # Take the last n_steps
    subset_history = history_df.iloc[-n_steps:].copy()
    
    for i in range(len(subset_history)):
        step_idx = i + 1
        current_data = subset_history.iloc[:step_idx]
        last_step = current_data.iloc[-1]
        
        # Create a frame
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Price and Trades
        window_start_idx = max(0, len(current_data) - 50)
        # Use integer positions for slicing to avoid DatetimeIndex loc issues
        # history_df['step'] contains the integer steps
        start_step = current_data.iloc[window_start_idx]['step']
        end_step = last_step['step']
        
        # Slice df (which has RangeIndex) using integer locations
        # But wait, df in main has DatetimeIndex. Let's use iloc if we have the index.
        # Find integer locations in df that match the steps
        # Actually, df.iloc[start_step:end_step+1] is safest if steps are continuous.
        plot_df = df.iloc[int(start_step):int(end_step)+1]
        
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='Price'), row=1, col=1)
        
        # Equity Curve in frame
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['net_worth'], name="Equity", line=dict(color='cyan')), row=2, col=1)
        
        # Add annotation for action
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_text = action_map.get(int(last_step['action']), "UNKNOWN")
        color = "white"
        if action_text == "BUY": color = "lime"
        if action_text == "SELL": color = "red"
        
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.95, text=f"ACTION: {action_text}", showarrow=False, font=dict(size=18, color=color), bgcolor="black", row=1, col=1)
        fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.95, text=f"NET WORTH: ${last_step['net_worth']:.2f}", showarrow=False, font=dict(size=14, color="cyan"), bgcolor="black", xanchor="right", row=1, col=1)

        fig.update_layout(template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, height=600, width=800, margin=dict(l=10, r=10, t=30, b=10))
        
        # Convert Plotly fig to image
        img_bytes = fig.to_image(format="png")
        frames.append(imageio.v2.imread(io.BytesIO(img_bytes)))
        
    imageio.mimsave(filename, frames, fps=5)
    print(f"Animated trading GIF saved to {filename}")


def run_prediction_audit(history_df):
    """PhDs love confusion matrices and calibration."""
    H = 12 # 12 hours later
    history_df['future_price'] = history_df['price'].shift(-H)
    history_df['target'] = (history_df['future_price'] > history_df['price']).astype(int) 
    
    audit_df = history_df[history_df['action'].isin([1, 2])].dropna().copy()
    audit_df['pred'] = audit_df['action'].map({1: 1, 2: 0})
    
    if not audit_df.empty:
        cm = confusion_matrix(audit_df['target'], audit_df['pred'])
        print("\n" + "="*60)
        print("         PREDICTION AUDIT (Action vs. 12H Forward Return)")
        print("="*60)
        print(f"Confusion Matrix:\n{cm}")
        print("\nClassification Report:")
        print(classification_report(audit_df['target'], audit_df['pred'], target_names=['Price Down', 'Price Up']))
        print("="*60)


def plot_trades(df, trades_df):
    """Generates an interactive Plotly chart of the trades."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], mode='lines', line=dict(color='rgba(255, 229, 153, 0.5)'), name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Lower Band', fill='tonexty'))

    if not trades_df.empty:
        buy_indices = [i for i in trades_df['entry_step'] if i < len(df)]
        sell_indices = [i for i in trades_df['exit_step'] if i < len(df)]
        buy_signals = df.iloc[buy_indices]
        sell_signals = df.iloc[sell_indices]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))
        
    fig.update_layout(title='Bitcoin Trading Bot Backtest: The Champion Model',
                      xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False, template='plotly_dark')
    try:
        fig.show()
    except Exception as e:
        print(f"Could not show plot: {e}")


def generate_tradingview_script(df, trades_df, filename="tradingview_strategy.pine"):
    """
    Generates a Pine Script™ file to visualize trades in TradingView.
    """
    if trades_df.empty:
        print("No trades to generate script for.")
        return

    buy_times = df.index[trades_df['entry_step']].tolist()
    sell_times = df.index[trades_df['exit_step']].tolist()

    buy_timestamps_ms = [int(t.timestamp() * 1000) for t in buy_times]
    sell_timestamps_ms = [int(t.timestamp() * 1000) for t in sell_times]

    pine_script_content = f"""
//@version=5
strategy("RL Agent Backtest", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.05)

var long_signals = array.from({', '.join(map(str, buy_timestamps_ms))})
var exit_signals = array.from({', '.join(map(str, sell_timestamps_ms))})

is_long_signal = array.includes(long_signals, time)
is_exit_signal = array.includes(exit_signals, time)

if (is_long_signal)
    strategy.entry("Long", strategy.long)
if (is_exit_signal)
    strategy.close("Long", comment="Exit")

plotshape(series=is_long_signal, title="Buy Signal", location=location.belowbar, color=color.new(color.green, 0), style=shape.labelup, text="BUY", textcolor=color.new(color.white, 0), size=size.small)
plotshape(series=is_exit_signal, title="Sell Signal", location=location.abovebar, color=color.new(color.red, 0), style=shape.labeldown, text="SELL", textcolor=color.new(color.white, 0), size=size.small)
"""

    with open(filename, "w") as f:
        f.write(pine_script_content)
    print(f"\nSuccessfully generated TradingView Pine Script: '{filename}'")
    print("  Instructions: Copy the content of this file into the TradingView Pine Editor.")


if __name__ == '__main__':
    # Set to True to retrain the final champion model. 
    # Set to False to load the last saved champion and just run the backtest.
    to_train = True

    print("Loading, preprocessing data for the final CHAMPION strategy (1H timeframe)...")
    train_df, test_df = preprocess_data('data/BTC-1m.csv')
    
    best_params = { 'learning_rate': 0.0006411154386870323, 'n_steps': 256, 'gamma': 0.9003194619328253 }
    best_lookback_window = 40
    
    MODEL_PATH = "ppo_trading_bot_1h_CHAMPION.zip"

    if to_train:
        print(f"Creating CHAMPION training environment with lookback: {best_lookback_window}...")
        train_env = TradingEnv(df=train_df, lookback_window=best_lookback_window)
        train_env = Monitor(train_env)
        
        TRAIN_TIMESTEPS = 10_000_000 
        print(f"Training CHAMPION agent for {TRAIN_TIMESTEPS} timesteps...")
        model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log="./ppo_champion_tensorboard/", device='cpu', **best_params)
        model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=True)
        print("Training finished.")
        print(f"Saving CHAMPION model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        
        print(f"\nCreating testing environment for final backtest...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        trades_df, history_df = run_detailed_backtest(test_env, model)

    else: 
        print(f"Loading pre-trained model from {MODEL_PATH}...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        model = PPO.load(MODEL_PATH, env=test_env)
        
        print("\nRunning backtest on loaded CHAMPION model...")
        trades_df, history_df = run_detailed_backtest(test_env, model)

    if not trades_df.empty: 
        print("\nGenerating Professional Analytics & Portfolio Dashboard...")
        metrics = calculate_advanced_metrics(history_df, trades_df, test_env.initial_balance)
        
        print("\n--- V. PhD Application Portfolio Metrics ---")
        print(f"Sortino Ratio:           {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:            {metrics['calmar_ratio']:.2f}")
        print(f"CAGR:                    {metrics['cagr']*100:.2f}%")
        print(f"Benchmark Return:        {metrics['benchmark_return']*100:.2f}%")
        print(f"Alpha (vs Benchmark):    {(metrics['total_return'] - metrics['benchmark_return'])*100:.2f}%")
        
        run_prediction_audit(history_df)
        create_professional_dashboard(test_df, history_df, trades_df, metrics)
        generate_trading_gif(test_df, history_df, n_steps=30)
        generate_tradingview_script(test_df, trades_df)