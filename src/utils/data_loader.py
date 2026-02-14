import pandas as pd
import pandas_ta as ta
import numpy as np

def preprocess_data(filepath):
    """Loads data, resamples to 1H, and correctly normalizes."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s')
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
