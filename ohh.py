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
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import warnings
import torch
from sklearn.metrics import confusion_matrix
import os
import matplotlib
matplotlib.use('Agg') # Prevent Tcl_AsyncDelete thread errors
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


class TradingEnv(gym.Env):
    """
    The final, refined environment to teach the agent to
    "cut losses short and let winners run."
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=100, lookback_window=40):
        super(TradingEnv, self).__init__()
        self.df = df.dropna()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_fee_percent = 0.0005 

        # Vectorization Optimization: Pre-calculate NumPy views
        self.features = ['price_vs_mavg', 'bandwidth', 'rsi', 'macd', 'stoch_k']
        self.df_values = self.df[self.features].values.astype(np.float32)
        self.close_prices = self.df['Close'].values.astype(np.float32)
        
        self.num_features = len(self.features) + 2 # + in_position, unrealized_profit
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback_window, self.num_features), dtype=np.float32
        )
        self.reset()

    def _get_obs(self):
        start = max(self.current_step - self.lookback_window + 1, 0)
        end = self.current_step + 1
        
        # Fast NumPy slicing instead of pandas
        obs_slice = self.df_values[start:end]
        
        # Calculate dynamic features
        in_position_flag = 1.0 if self.position_open else 0.0
        current_price = self.close_prices[self.current_step]
        unrealized_profit = (current_price - self.entry_price) / self.entry_price if self.position_open else 0.0
        
        # Create observation with dynamic features
        # Add columns for in_position and unrealized_profit
        rows = obs_slice.shape[0]
        dynamic_features = np.zeros((rows, 2), dtype=np.float32)
        dynamic_features[:, 0] = in_position_flag
        dynamic_features[:, 1] = unrealized_profit
        
        full_obs = np.hstack([obs_slice, dynamic_features])
        
        # Padding if necessary
        if rows < self.lookback_window:
            padding = np.zeros((self.lookback_window - rows, self.num_features), dtype=np.float32)
            full_obs = np.vstack([padding, full_obs])
            
        return full_obs.astype(np.float32)

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
        current_price = self.df['Close'].iloc[self.current_step]
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
        
        self.history.append({'step': self.current_step, 'net_worth': self.net_worth})

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
    df['date'] = pd.to_datetime(df['Timestamp'],unit='s')
    df = df.sort_values('date').set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    aggregation = { 'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum' }
    df = df.resample('1h').agg(aggregation).dropna()

    df.ta.bbands(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    
    rename_dict = {}
    # print(df.head())
    for col in df.columns:
        if 'BBL' in col: rename_dict[col] = 'lower_band'
        if 'BBM' in col: rename_dict[col] = 'middle_band'
        if 'BBU' in col: rename_dict[col] = 'upper_band'
    df = df.rename(columns=rename_dict)

    df = df.rename(columns={"RSI_14": "rsi", "MACD_12_26_9": "macd", "STOCHk_14_3_3": "stoch_k"})
    
    epsilon = 1e-10
    df['price_vs_mavg'] = (df['Close'] - df['middle_band']) / (df['upper_band'] - df['lower_band'] + epsilon)
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


def run_detailed_backtest(env, model, suffix="", check_improvement=False, previous_best=-np.inf):
    """
    Runs a backtest and captures exhaustive data for professional analytics.
    Returns: (trades, net_worth)
    """
    obs, info = env.reset()
    num_steps = len(env.df) - env.lookback_window
    
    agent_data = []
    
    device = model.device
    
    # for _ in tqdm(range(num_steps), desc="Backtesting"):
    for _ in tqdm(range(num_steps), desc="Backtesting", disable=True):
        # Get action probabilities for calibration analysis (kept for dashboard)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]
            
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        agent_data.append({
            'step': env.current_step,
            'action': action,
            'probs': probs,
            'net_worth': env.net_worth,
            'price': env.close_prices[env.current_step]
        })
        
        if done or truncated: break

    if env.position_open:
        exit_price = env.close_prices[env.current_step]
        position_value = env.last_balance * (exit_price / env.entry_price)
        sell_fee = position_value * env.transaction_fee_percent
        env.balance = position_value - sell_fee
        env.net_worth = env.balance
        net_profit = env.balance - env.last_balance
        net_profit_percent = (net_profit / env.last_balance) * 100 if env.last_balance > 0 else 0
        trade = {'entry_step': env.entry_step, 'exit_step': env.current_step,
                 'entry_price': env.entry_price, 'exit_price': exit_price, 'profit_%': net_profit_percent}
        env.trades.append(trade)

    # Conditional Saving Logic
    if check_improvement:
        if env.net_worth <= previous_best:
            # Not an improvement, skip reporting
            return env.trades, env.net_worth

    analyzer = PerformanceAnalyzer(env, agent_data, suffix=suffix)
    analyzer.generate_report()
    
    return env.trades, env.net_worth

class PerformanceAnalyzer:
    def __init__(self, env, agent_data, suffix=""):
        self.env = env
        self.df = env.df
        self.trades = env.trades
        self.agent_data = pd.DataFrame(agent_data)
        self.suffix = f"_{suffix}" if suffix else ""
        self.output_dir = "backtest_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.history = pd.DataFrame(env.history).set_index('step')
        self.history.index = self.df.index[self.history.index]
        
        # Calculate Benchmark (Buy & Hold)
        self.history['benchmark_price'] = self.df['Close'].loc[self.history.index]
        self.history['benchmark_net_worth'] = self.env.initial_balance * (self.history['benchmark_price'] / self.history['benchmark_price'].iloc[0])
        
        # Calculate 5x Leverage Simulation
        self.history['returns'] = self.history['net_worth'].pct_change().fillna(0)
        self.history['leverage_5x_nw'] = self.env.initial_balance * (1 + self.history['returns'] * 5).cumprod()

    def calculate_metrics(self):
        rets = self.history['returns']
        trading_periods = 24 * 365
        
        # Sharpe Ratio
        sharpe = (rets.mean() / rets.std()) * np.sqrt(trading_periods) if rets.std() > 0 else 0
        
        # Sortino Ratio (Downside deviation)
        downside_std = rets[rets < 0].std()
        sortino = (rets.mean() / downside_std) * np.sqrt(trading_periods) if downside_std > 0 else 0
        
        rolling_sharpe = rets.rolling(window=24*30).apply(lambda x: (x.mean() / x.std()) * np.sqrt(trading_periods) if x.std() > 0 else 0)
        
        self.history['cum_max'] = self.history['net_worth'].cummax()
        self.history['drawdown'] = (self.history['net_worth'] - self.history['cum_max']) / self.history['cum_max']
        max_dd = self.history['drawdown'].min()
        
        # Calmar Ratio
        calmar = (rets.mean() * trading_periods) / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
            'rolling_sharpe': rolling_sharpe
        }

    def generate_report(self):
        metrics = self.calculate_metrics()
        
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append(" " * 25 + "ADVANCED RESEARCH PERFORMANCE REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Final Net Worth:         ${self.env.net_worth:,.2f}")
        report_lines.append(f"Total Strategy Return:   {((self.env.net_worth - self.env.initial_balance) / self.env.initial_balance * 100):.2f}%")
        report_lines.append(f"Benchmark (B&H) Return:  {((self.history['benchmark_net_worth'].iloc[-1] - self.env.initial_balance) / self.env.initial_balance * 100):.2f}%")
        report_lines.append("-" * 80)
        report_lines.append(f"Annualized Sharpe:       {metrics['sharpe']:.2f}")
        report_lines.append(f"Annualized Sortino:      {metrics['sortino']:.2f}")
        report_lines.append(f"Calmar Ratio:            {metrics['calmar']:.2f}")
        report_lines.append(f"Max Drawdown:            {metrics['max_dd']*100:.2f}%")
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            wins = trades_df[trades_df['profit_%'] > 0]
            report_lines.append("-" * 80)
            report_lines.append(f"Total Trades:            {len(trades_df)}")
            report_lines.append(f"Win Rate:                {(len(wins) / len(trades_df) * 100):.2f}%")
            report_lines.append(f"Profit Factor:           {(wins['profit_%'].sum() / abs(trades_df[trades_df['profit_%'] <= 0]['profit_%'].sum())):.2f}")
            
            report_lines.append("\n" + "-"*35 + " TRADE BREAKDOWN " + "-"*35)
            report_lines.append(trades_df[['entry_step', 'exit_step', 'profit_%']].sort_values(by='profit_%', ascending=False).head(5).to_string(index=False))
            report_lines.append("...")
            report_lines.append(trades_df[['entry_step', 'exit_step', 'profit_%']].sort_values(by='profit_%', ascending=False).tail(5).to_string(index=False))
        
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        # print(report_content)
        
        report_path = os.path.join(self.output_dir, f"performance_report{self.suffix}.txt")
        with open(report_path, "w") as f:
            f.write(report_content)
        # tqdm.write(f"[SUCCESS] Text Report saved: '{report_path}'")

        self.create_visual_dashboard(metrics)
        self.save_agent_profile_image(metrics)

    def save_agent_profile_image(self, metrics):
        """Generates a high-resolution summary image for portfolios."""
        categories = ['Return Intensity', 'Risk-Adjusted (Sharpe)', 'Downside Risk (Sortino)', 'Robustness (Inv. DD)', 'Win Consistency', 'Profit Efficiency']
        
        # Recalculate scores for radar
        ret_val = (self.env.net_worth - self.env.initial_balance) / self.env.initial_balance
        score_return = min(max(ret_val / 0.5, 0), 1)
        score_sharpe = min(max(metrics['sharpe'] / 3.0, 0), 1)
        score_sortino = min(max(metrics['sortino'] / 3.0, 0), 1)
        score_drawdown = 1.0 - min(abs(metrics['max_dd']) / 0.5, 1)
        
        trades_df = pd.DataFrame(self.trades) if self.trades else None
        if trades_df is not None:
            win_rate = len(trades_df[trades_df['profit_%'] > 0]) / len(trades_df)
            score_win_rate = min(max((win_rate - 0.45) / 0.15, 0), 1)
            profit_factor = trades_df[trades_df['profit_%'] > 0]['profit_%'].sum() / abs(trades_df[trades_df['profit_%'] <= 0]['profit_%'].sum())
            score_profit_factor = min(max((profit_factor - 1.0) / 0.5, 0), 1)
        else:
            win_rate, profit_factor = 0, 0
            score_win_rate, score_profit_factor = 0, 0

        values = [score_return, score_sharpe, score_sortino, score_drawdown, score_win_rate, score_profit_factor]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        plt.style.use('dark_background')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')

        plt.xticks(angles[:-1], categories, color='cyan', size=10)
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='cyan')
        ax.fill(angles, values, 'cyan', alpha=0.3)
        ax.set_ylim(0, 1)
        
        title_str = "Agent Capability Profile: PhD Portfolio Summary"
        plt.title(title_str, size=16, color='white', y=1.1)
        
        # Add summary text Box
        textstr = '\n'.join((
            f"Strategy: PPO Reinforcement Learning",
            f"Total Return: {(ret_val*100):.2f}%",
            f"Sharpe Ratio: {metrics['sharpe']:.2f}",
            f"Max Drawdown: {(metrics['max_dd']*100):.2f}%",
            f"Win Rate: {(win_rate*100):.2f}%",
            f"Profit Factor: {profit_factor:.2f}"
        ))
        
        props = dict(boxstyle='round', facecolor='indigo', alpha=0.5)
        ax.text(1.3, 1.1, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, color='white')

        profile_path = os.path.join(self.output_dir, f"agent_profile{self.suffix}.png")
        profile_path = os.path.join(self.output_dir, f"agent_profile{self.suffix}.png")
        plt.savefig(profile_path, dpi=300, bbox_inches='tight')
        plt.close()
        # tqdm.write(f"[SUCCESS] Summary Profile Image saved: '{profile_path}'")

    def create_visual_dashboard(self, metrics):
        # Calculate scores for Radar Chart (Normalized 0-1)
        # Returns: 100% Return -> 1.0 (Log scale for better visualization)
        ret_val = (self.env.net_worth - self.env.initial_balance) / self.env.initial_balance
        score_return = min(max(ret_val / 0.5, 0), 1) # 50% is 'full'
        score_sharpe = min(max(metrics['sharpe'] / 3.0, 0), 1) # 3.0 is institutional grade
        score_sortino = min(max(metrics['sortino'] / 3.0, 0), 1)
        score_drawdown = 1.0 - min(abs(metrics['max_dd']) / 0.5, 1) # 50% DD is 'empty'
        
        trades_df = pd.DataFrame(self.trades) if self.trades else None
        if trades_df is not None:
            win_rate = len(trades_df[trades_df['profit_%'] > 0]) / len(trades_df)
            score_win_rate = min(max((win_rate - 0.45) / 0.15, 0), 1) # 45% to 60% range
            profit_factor = trades_df[trades_df['profit_%'] > 0]['profit_%'].sum() / abs(trades_df[trades_df['profit_%'] <= 0]['profit_%'].sum())
            score_profit_factor = min(max((profit_factor - 1.0) / 0.5, 0), 1) # 1.0 to 1.5 range
        else:
            score_win_rate = 0
            score_profit_factor = 0

        # Create Layout
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=(
                "Agent Ability Profile (Radar)", "Strategic Summary Stats",
                "Equity Curve vs Benchmark", "Drawdown Profile (%)",
                "Daily Returns Distribution", "Rolling Sharpe Ratio (30D)",
                "Action vs Next-Bar Reality (Confusion)", "Model Confidence Calibration",
                "1x vs 5x Leverage Stress Test", "Trade Profit Distribution",
                "Cumulative Return %", "Consistency Analysis (Rolling PnL)"
            ),
            vertical_spacing=0.06,
            specs=[[{"type": "polar"}, {"type": "table"}], 
                   [{"secondary_y": True}, {}], 
                   [{}, {}], 
                   [{}, {}], 
                   [{}, {}], 
                   [{}, {}]]
        )

        # 0. Radar Chart
        fig.add_trace(go.Scatterpolar(
            r=[score_return, score_sharpe, score_sortino, score_drawdown, score_win_rate, score_profit_factor],
            theta=['Return Intensity', 'Risk-Adjusted (Sharpe)', 'Downside Risk (Sortino)', 'Robustness (Inv. DD)', 'Win Consistency', 'Profit Efficiency'],
            fill='toself',
            name='Agent Capability',
            line_color='cyan'
        ), row=1, col=1)

        # 0. Summary Table
        summary_stats = [
            ['Methodology', 'PPO Reinforcement Learning'],
            ['Timeframe', '1-Hour (H1) BTC/USD'],
            ['Total Return', f"{(ret_val*100):.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe']:.2f}"],
            ['Max Drawdown', f"{(metrics['max_dd']*100):.2f}%"],
            ['Profit Factor', f"{profit_factor:.2f}" if trades_df is not None else "N/A"],
            ['Win Rate', f"{(win_rate*100):.2f}%" if trades_df is not None else "N/A"]
        ]
        fig.add_trace(go.Table(
            header=dict(values=['Dimension', 'Metric'], fill_color='indigo', align='left', font=dict(color='white', size=12)),
            cells=dict(values=[[s[0] for s in summary_stats], [s[1] for s in summary_stats]], fill_color='darkslateblue', align='left', font=dict(color='white', size=11))
        ), row=1, col=2)

        # 1. Equity Curve
        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['net_worth'], name="Strategy (1x)", line=dict(color='cyan', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['benchmark_net_worth'], name="Benchmark (B&H)", line=dict(color='gray', dash='dash')), row=2, col=1)

        # 2. Drawdown
        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['drawdown']*100, fill='tozeroy', name="Drawdown %", line=dict(color='rgba(255, 0, 0, 0.5)')), row=2, col=2)
        # Highlight Max Drawdown
        max_dd_idx = self.history['drawdown'].idxmin()
        fig.add_trace(go.Scatter(x=[max_dd_idx], y=[self.history['drawdown'].min()*100], mode='markers', marker=dict(color='yellow', size=12, symbol='x'), name="Max DD"), row=2, col=2)

        # 3. Returns Distribution
        daily_rets = self.history['returns'].resample('D').sum()
        fig.add_trace(go.Histogram(x=daily_rets, name="Daily Returns", marker_color='royalblue', opacity=0.7, nbinsx=50), row=3, col=1)

        # 4. Rolling Sharpe
        fig.add_trace(go.Scatter(x=self.history.index, y=metrics['rolling_sharpe'], name="Rolling Sharpe", line=dict(color='orange')), row=3, col=2)

        # 5. Confusion Matrix
        # Define Ground Truth: 1=Next bar up, 2=Next bar down, 0=Flat
        self.agent_data['next_ret'] = self.agent_data['price'].pct_change().shift(-1)
        self.agent_data['truth'] = self.agent_data['next_ret'].apply(lambda x: 1 if x > 0.0001 else (2 if x < -0.0001 else 0))
        
        # Clean data for Confusion Matrix to avoid "unknown targets" error
        cm_df = self.agent_data[['truth', 'action']].dropna()
        y_true = cm_df['truth'].astype(int)
        y_pred = cm_df['action'].astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        # Normalize CM for better visualization
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig.add_trace(go.Heatmap(z=cm_norm, x=['Hold', 'Buy', 'Sell'], y=['Reality: Flat', 'Reality: Up', 'Reality: Down'], colorscale='Plasma', name="CM"), row=4, col=1)

        # 6. Confidence Calibration (Box plot per action)
        self.agent_data['conf'] = self.agent_data['probs'].apply(lambda x: np.max(x))
        for act, label in zip([0, 1, 2], ['Hold', 'Buy', 'Sell']):
            act_data = self.agent_data[self.agent_data['action'] == act]
            fig.add_trace(go.Box(y=act_data['conf'], name=label, boxpoints='outliers'), row=4, col=2)

        # 7. Leverage Comparison
        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['net_worth'], name="1x NW", line=dict(color='cyan')), row=5, col=1)
        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['leverage_5x_nw'], name="5x NW", line=dict(color='rgba(255, 0, 255, 0.4)')), row=5, col=1)

        # 8. Trade Profit Distribution
        if self.trades:
            profit_data = [t['profit_%'] for t in self.trades]
            fig.add_trace(go.Box(x=profit_data, name="Trade PnL %", marker_color='lime'), row=5, col=2)

        # 9. Cumulative Return %
        cum_pnl = (self.history['net_worth'] / self.env.initial_balance - 1) * 100
        fig.add_trace(go.Scatter(x=self.history.index, y=cum_pnl, name="Cum PnL %", line=dict(color='lime')), row=6, col=1)

        # 10. Rolling Win Rate (Moving average of profitability)
        if self.trades:
            trades_df['win'] = (trades_df['profit_%'] > 0).astype(int)
            trades_df['rolling_win_rate'] = trades_df['win'].rolling(window=10).mean()
            # Map steps back to dates for plotting
            trades_df['exit_date'] = self.df.index[trades_df['exit_step']]
            fig.add_trace(go.Scatter(x=trades_df['exit_date'], y=trades_df['rolling_win_rate'], name="Win Rate (10-Trade MA)", line=dict(color='gold')), row=6, col=2)

        fig.update_layout(height=2200, width=1200, title_text=f"Bitcoin RL Trading: Step {self.suffix.replace('_', '')} Portfolio", template="plotly_dark", showlegend=True)
        dashboard_path = os.path.join(self.output_dir, f"performance_dashboard{self.suffix}.html")
        dashboard_path = os.path.join(self.output_dir, f"performance_dashboard{self.suffix}.html")
        fig.write_html(dashboard_path)
        fig.write_html(dashboard_path)
        # tqdm.write(f"[SUCCESS] Advanced Dashboard generated: '{dashboard_path}'")
        # tqdm.write(f"           - Methodology: 1H Timeframe, PPO Algorithm, Engineered Technical Features")
        # tqdm.write(f"           - Research Context: High-frequency decision making under transaction cost constraints.")


def plot_trades(df, trades):
    """Generates an interactive Plotly chart of the trades."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], mode='lines', line=dict(color='rgba(255, 229, 153, 0.5)'), name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Lower Band', fill='tonexty'))

    if trades:
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            buy_indices = [i for i in trades_df['entry_step'] if i < len(df)]
            sell_indices = [i for i in trades_df['exit_step'] if i < len(df)]
            buy_signals = df.iloc[buy_indices]
            sell_signals = df.iloc[sell_indices]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy Signal'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))
        
    fig.update_layout(title='Bitcoin Trading Bot Backtest: The Champion Model',
                      xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.show()


def generate_tradingview_script(df, trades, filename="tradingview_strategy.pine"):
    """
    Generates a Pine Script™ file to visualize trades in TradingView.
    """
    if not trades:
        print("No trades to generate script for.")
        return

    trades_df = pd.DataFrame(trades)
    buy_times = df.index[trades_df['entry_step']].tolist()
    sell_times = df.index[trades_df['exit_step']].tolist()

    buy_timestamps_ms = [int(t.timestamp() * 1000) for t in buy_times]
    sell_timestamps_ms = [int(t.timestamp() * 1000) for t in sell_times]

    pine_script_content = f"""
//@version=5
strategy("RL Agent Backtest", overlay=true, initial_capital={trades_df.iloc[0].get('initial_balance', 10000)}, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.05)

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


class PeriodicBacktestCallback(BaseCallback):
    """
    Callback for running backtests and saving reports periodically.
    """
    def __init__(self, test_df, lookback_window, check_freq=5000, verbose=0):
        super(PeriodicBacktestCallback, self).__init__(verbose)
        self.test_df = test_df
        self.lookback_window = lookback_window
        self.check_freq = check_freq
        self.best_net_worth = -np.inf # Track best performance

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # tqdm.write(f"\n[CALLBACK] Running intermediate backtest at step {self.num_timesteps}...")
            # Create a fresh environment for the backtest
            test_env = TradingEnv(df=self.test_df, lookback_window=self.lookback_window)
            suffix = f"step_{self.num_timesteps}"
            
            # Run backtest using the current model state
            # Pass check_improvement=True so we only save if we beat the previous best
            trades, net_worth = run_detailed_backtest(
                test_env, 
                self.model, 
                suffix=suffix, 
                check_improvement=True, 
                previous_best=self.best_net_worth
            )
            
            # If we found a new best, update our record
            if net_worth > self.best_net_worth:
                self.best_net_worth = net_worth
                # Optional: could print a small notification even in silent mode? 
                # User asked for no output, so keeping it silent.
                
        return True


if __name__ == '__main__':
    # Set to True to retrain the final champion model. 
    # Set to False to load the last saved champion and just run the backtest.
    to_train = True

    # print("Loading, preprocessing data for the final CHAMPION strategy (1H timeframe)...")
    train_df, test_df = preprocess_data('data/BTC-1m.csv')
    
    best_params = { 'learning_rate': 0.0006411154386870323, 'n_steps': 256, 'gamma': 0.9003194619328253 }
    best_lookback_window = 40
    
    MODEL_PATH = "ppo_trading_bot_1h_CHAMPION.zip"

    if to_train:
        # print(f"Creating CHAMPION training environment with lookback: {best_lookback_window}...")
        train_env = TradingEnv(df=train_df, lookback_window=best_lookback_window)
        train_env = Monitor(train_env)
        
        TRAIN_TIMESTEPS = 100_000_000 # Restored original value
        
        # Explicit device check
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu" # User requested CPU due to better performance for this specific model/batch size
        # print(f"Using device: {device}")
        
        # print(f"Training CHAMPION agent for {TRAIN_TIMESTEPS} timesteps...")
        
        # Setup the callback
        backtest_callback = PeriodicBacktestCallback(test_df=test_df, lookback_window=best_lookback_window, check_freq=100000)
        
        model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log="./ppo_champion_tensorboard/", device=device, **best_params)
        model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=backtest_callback, progress_bar=True)
        # print("Training finished.")
        # print(f"Saving CHAMPION model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        
        print(f"\nCreating testing environment for final backtest...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        trades, final_nw = run_detailed_backtest(test_env, model)

    else: 
        print(f"Loading pre-trained model from {MODEL_PATH}...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        model = PPO.load(MODEL_PATH, env=test_env)
        
        print("\nRunning backtest on loaded CHAMPION model...")
        trades, final_nw = run_detailed_backtest(test_env, model)

    if trades: 
        print("Generating interactive plot of CHAMPION trades...")
        # You can choose which visualization to run, or run both
        plot_trades(test_df, trades) 
        generate_tradingview_script(test_df, trades)