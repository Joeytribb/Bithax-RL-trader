import pandas as pd
import numpy as np

# Step 1: Try importing pandas_ta
try:
    import pandas_ta as ta
    print("✅ pandas-ta successfully imported")
except ImportError as e:
    print("❌ pandas-ta is not installed or not working")
    raise e

# Step 2: Create synthetic OHLCV data
np.random.seed(42)
rows = 100

df = pd.DataFrame({
    "open": np.random.random(rows) * 100,
    "high": np.random.random(rows) * 100,
    "low": np.random.random(rows) * 100,
    "close": np.random.random(rows) * 100,
    "volume": np.random.randint(1000, 10000, rows)
})

# Step 3: Calculate indicators
df["SMA_14"] = ta.sma(df["close"], length=14)
df["RSI_14"] = ta.rsi(df["close"], length=14)
macd = ta.macd(df["close"])

df = pd.concat([df, macd], axis=1)

# Step 4: Basic validation
if df["SMA_14"].isnull().all():
    print("❌ SMA failed")
else:
    print("✅ SMA computed")

if df["RSI_14"].isnull().all():
    print("❌ RSI failed")
else:
    print("✅ RSI computed")

if macd is None or macd.isnull().all().all():
    print("❌ MACD failed")
else:
    print("✅ MACD computed")

print("\nPreview:")
print(df.tail())
