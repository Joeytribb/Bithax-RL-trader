#debugged perfected version
import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)


class TradingEnv(gym.Env):
    """
    The final, refined environment from the enhanced version,
    ported to the OG script for stability.
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
    """Loads data, resamples to 1h, and correctly normalizes indicators."""
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    
    # Robust timestamp handling
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Fallback if neither found (e.g. standard CCXT format might use 'time')
        df['date'] = pd.to_datetime(df.iloc[:, 0])
        
    df = df.sort_values('date').set_index('date')
    
    # Standardize columns to expected capitalization for technical indicators if needed
    # but pandas_ta handles any casing if you specify 'close' etc.
    aggregation = { 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum' }
    df = df.resample('1h').agg(aggregation).dropna()

    # Explicitly calculate indicators using the robust direct-access method
    bb_df = df.ta.bbands(length=20)
    rsi_ser = df.ta.rsi(length=14)
    macd_df = df.ta.macd(fast=12, slow=26)
    stoch_df = df.ta.stoch(k=14, d=3)
    
    # Verification and assignment
    df['lower_band'] = bb_df.iloc[:, 0].squeeze()
    df['middle_band'] = bb_df.iloc[:, 1].squeeze()
    df['upper_band'] = bb_df.iloc[:, 2].squeeze()
    df['rsi'] = rsi_ser.squeeze()
    df['macd'] = macd_df.iloc[:, 0].squeeze()
    df['stoch_k'] = stoch_df.iloc[:, 0].squeeze()
    
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

    # Final Liquidation
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
        return trades

    trades_df = pd.DataFrame(trades)
    trades_df['entry_date'] = env.df.index[trades_df['entry_step']]
    trades_df['exit_date'] = env.df.index[trades_df['exit_step']]
    trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']
    
    print("\n--- I. Core Performance Metrics ---")
    total_net_profit = final_balance - initial_balance
    total_net_profit_pct = (total_net_profit / initial_balance) * 100
    print(f"Total Net Profit:        ${total_net_profit:,.2f} ({total_net_profit_pct:.2f}%)")

    history_df = pd.DataFrame(env.history).set_index('step')
    history_df.index = env.df.index[history_df.index]
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
    print(f"Win Rate:                {(len(wins) / len(trades_df) * 100 if len(trades_df)>0 else 0):.2f}%")
    if not losses.empty and abs(losses.sum()) > 0:
        print(f"Profit Factor:           {(wins.sum() / abs(losses.sum())):.2f}")
    
    print(f"Best Trade:              {profits.max():.2f}%")
    print(f"Worst Trade:             {profits.min():.2f}%")
    print(f"Average Trade P/L:       {profits.mean():.2f}%")
    
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
    return trades 


def plot_trades(df, trades):
    """Generates an interactive Plotly chart of the trades."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], mode='lines', line=dict(color='rgba(255, 229, 153, 0.5)'), name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', line=dict(color='rgba(173, 204, 255, 0.5)'), name='Lower Band', fill='tonexty'))

    if trades:
        trades_df = pd.DataFrame(trades)
        buy_indices = [i for i in trades_df['entry_step'] if i < len(df)]
        sell_indices = [i for i in trades_df['exit_step'] if i < len(df)]
        buy_signals = df.iloc[buy_indices]
        sell_signals = df.iloc[sell_indices]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))
        
    fig.update_layout(title='Bitcoin Trading Bot Backtest: OG Strategy Corrected',
                      xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.show()


if __name__ == '__main__':
    # Production values
    to_train = False
    
    print("Loading, preprocessing data (with robust restoration)...")
    train_df, test_df = preprocess_data('data/BTC-1m.csv')
    
    best_params = { 'learning_rate': 0.0006411154386870323, 'n_steps': 256, 'gamma': 0.9003194619328253 }
    best_lookback_window = 40
    
    MODEL_PATH = "ppo_trading_bot_1h_CHAMPION.zip"

    if to_train:
        print(f"Creating training environment with lookback: {best_lookback_window}...")
        train_env = TradingEnv(df=train_df, lookback_window=best_lookback_window)
        train_env = Monitor(train_env)
        
        TRAIN_TIMESTEPS = 600_000 
        print(f"Training agent for {TRAIN_TIMESTEPS} timesteps...")
        model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log="./ppo_trading_tensorboard/", device='cpu', **best_params)
        model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=True)
        print("Training finished.")
        print(f"Saving debugged model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        
        print(f"\nCreating testing environment ({len(test_df)} steps)...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        trades = run_detailed_backtest(test_env, model)

    else: 
        print(f"Loading pre-trained model from {MODEL_PATH}...")
        test_env = TradingEnv(df=test_df, lookback_window=best_lookback_window)
        model = PPO.load(MODEL_PATH, env=test_env)
        
        print("\nRunning backtest on loaded model...")
        trades = run_detailed_backtest(test_env, model)

    if trades: 
        print("Generating interactive plot of trades...")
        plot_trades(test_df, trades)