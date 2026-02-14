from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from src.env.trading_env import TradingEnv
from src.utils.reporting import run_detailed_backtest

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
