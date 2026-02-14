import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

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
