import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch
import warnings
import matplotlib

# Set matplotlib backend to avoid Tcl_AsyncDelete errors
matplotlib.use('Agg')

# Import custom modules
from src.utils.data_loader import preprocess_data
from src.env.trading_env import TradingEnv
from src.utils.callbacks import PeriodicBacktestCallback
from src.utils.reporting import run_detailed_backtest, plot_trades, generate_tradingview_script

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # Set to True to retrain the final champion model. 
    # Set to False to load the last saved champion and just run the backtest.
    to_train = True

    # print("Loading, preprocessing data for the final CHAMPION strategy (1H timeframe)...")
    train_df, test_df = preprocess_data('data/BTC-1m.csv')
    
    best_params = { 'learning_rate': 0.0006411154386870323, 'n_steps': 256, 'gamma': 0.9003194619328253 }
    best_lookback_window = 40
    
    MODEL_PATH = "models/ppo_trading_bot_1h_CHAMPION.zip" # Updated path

    if to_train:
        # print(f"Creating CHAMPION training environment with lookback: {best_lookback_window}...")
        train_env = TradingEnv(df=train_df, lookback_window=best_lookback_window)
        train_env = Monitor(train_env)
        
        TRAIN_TIMESTEPS = 100_000_000 
        
        # Explicit device check
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu" # User requested CPU
        # print(f"Using device: {device}")
        
        # print(f"Training CHAMPION agent for {TRAIN_TIMESTEPS} timesteps...")
        
        # Setup the callback
        backtest_callback = PeriodicBacktestCallback(test_df=test_df, lookback_window=best_lookback_window, check_freq=100000)
        
        # Update tensorboard log path
        model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log="results/tensorboard/ppo_champion_tensorboard/", device=device, **best_params)
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
