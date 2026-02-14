import pandas as pd
import numpy as np
import pandas_ta as ta

# Scikit-learn for classical models and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 1. LOAD AND PREPARE DATA ---
print("Step 1: Loading and Preparing Data...")
try:
    df = pd.read_csv('data/BTC-1m.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
except FileNotFoundError:
    print("Error: 'data/BTC-1m.csv' not found. Please ensure the file exists.")
    exit()

df = df.iloc[int(len(df) * 0.5):]
df_resampled = df.resample('30T').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()
print(f"Data Resampled. Shape: {df_resampled.shape}")

# --- 2. FEATURE ENGINEERING ---
print("\nStep 2: Engineering Features...")
df_resampled.ta.rsi(length=14, append=True)
df_resampled.ta.macd(fast=12, slow=26, signal=9, append=True)
df_resampled.ta.bbands(length=20, append=True)
df_resampled.ta.atr(length=14, append=True)
df_resampled.ta.sma(length=50, append=True)
df_resampled.ta.sma(length=200, append=True)
df_resampled['price_change_1'] = df_resampled['Close'].pct_change(1)
df_resampled['price_change_4'] = df_resampled['Close'].pct_change(4)
df_resampled.dropna(inplace=True)
print(f"Features created. Shape after dropping NaNs: {df_resampled.shape}")

# --- 3. TRIPLE BARRIER LABELING ---
print("\nStep 3: Applying Triple Barrier Labeling...")
profit_target = 0.01
stop_loss = -0.005
time_barrier = 16 # 16 candles * 30 mins = 8 hours

def get_label(df_slice, entry_price):
    for i in range(len(df_slice)):
        if (df_slice['High'].iloc[i] - entry_price) / entry_price >= profit_target:
            return 1
        if (df_slice['Low'].iloc[i] - entry_price) / entry_price <= stop_loss:
            return 0
    return 0

labels = []
prices = df_resampled['Close']
for i in range(len(prices) - time_barrier):
    future_slice = df_resampled.iloc[i+1 : i+1+time_barrier]
    labels.append(get_label(future_slice, prices.iloc[i]))

X = df_resampled.iloc[:len(labels)].copy()
y = pd.Series(labels, index=X.index)
print(f"Labeling complete. Label distribution:\n{y.value_counts(normalize=True)}")

# --- 4. TRAIN & EVALUATE CLASSICAL MODELS ---
print("\nStep 4: Training and Evaluating Classical Models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report
    print(classification_report(y_test, y_pred))

# --- 5. PYTORCH LSTM IMPLEMENTATION ---
print("\nStep 5: PyTorch LSTM Implementation...")

# A. Data Preparation for PyTorch
sequence_length = 30 # Use last 30 candles (15 hours)

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_seq).float()
y_train_tensor = torch.from_numpy(y_train_seq).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test_seq).float()
y_test_tensor = torch.from_numpy(y_test_seq).float().view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# B. Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Use the output of the last time step
        out = self.sigmoid(out)
        return out

# Model parameters
input_size = X_train_seq.shape[2]
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

# C. Training Loop
print("--- Training PyTorch LSTM ---")
for epoch in range(num_epochs):
    lstm_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# D. Evaluation Loop
print("\n--- Evaluating PyTorch LSTM ---")
lstm_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = lstm_model(batch_X)
        preds = (outputs > 0.5).cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy().flatten())

report = classification_report(all_labels, all_preds, output_dict=True)
results["PyTorch LSTM"] = report
print(classification_report(all_labels, all_preds))


# --- 6. FINAL RESULTS COMPARISON ---
print("\n--- Model Comparison Summary ---")

summary = pd.DataFrame({
    "Model": list(results.keys()),
    "Precision (Win)": [results[m]['1']['precision'] for m in results],
    "Recall (Win)": [results[m]['1']['recall'] for m in results],
    "F1-Score (Win)": [results[m]['1']['f1-score'] for m in results],
    "Accuracy": [results[m]['accuracy'] for m in results]
}).sort_values(by="Precision (Win)", ascending=False)

print(summary.to_string())