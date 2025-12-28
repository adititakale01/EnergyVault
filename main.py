import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
SEQ_LEN = 50       # Length of one time-series window
HIDDEN_DIM = 10    # Compressed representation
EPOCHS = 50        # Fast training for demo
LR = 1e-3
BATCH_SIZE = 16

# --- 1. SIMULATE BATTERY DATA (The "Prototyping" Part) ---
# Simulates Voltage curves during charging
def generate_battery_data(n_cycles=200, mode='healthy'):
    data = []
    for i in range(n_cycles):
        # Time steps 0 to 100
        t = np.linspace(0, 10, SEQ_LEN)
        
        # Base Curve: Logarithmic growth (like battery charging)
        # Healthy: fast charge, stable max voltage
        # Faulty: slower charge, noisy, doesn't reach max voltage
        if mode == 'healthy':
            noise = np.random.normal(0, 0.02, SEQ_LEN)
            voltage = 3.2 + 1.0 * (1 - np.exp(-t)) + noise
        else:
            # Anomaly: "Aging" effect (higher resistance, noise)
            drift = np.random.uniform(0.1, 0.3) 
            noise = np.random.normal(0, 0.08, SEQ_LEN) # More noise
            voltage = 3.2 + (1.0 - drift) * (1 - np.exp(-t)) + noise
            
        data.append(voltage)
    return np.array(data, dtype=np.float32)

print("⚡ Generating Synthetic Test Bench Data...")
healthy_data = generate_battery_data(n_cycles=1000, mode='healthy')
faulty_data = generate_battery_data(n_cycles=200, mode='faulty')

# Reshape for PyTorch: [Samples, Sequence Length, Features(1)]
# We add a dimension for "Features" (Voltage)
X_train = torch.tensor(healthy_data).unsqueeze(-1)
X_test_faulty = torch.tensor(faulty_data).unsqueeze(-1)

# --- 2. BUILD THE MODEL (PyTorch LSTM Autoencoder) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x) # hidden: [1, batch, hidden_dim]
        # Repeat hidden state
        hidden_repeated = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        # Decode
        decoded, _ = self.decoder(hidden_repeated)
        # Reconstruct
        reconstruction = self.output_layer(decoded)
        return reconstruction

print("🧠 Initializing Deep Learning Model...")
model = LSTMAutoencoder(input_dim=1, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --- 3. TRAIN ON HEALTHY DATA ONLY ---
print("🏋️ Training (Self-Supervised)...")
model.train()
losses = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, X_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# --- 4. VALIDATION & VISUALIZATION ---
print("📊 Validating on Faulty Data...")
model.eval()

# Inference on Healthy vs Faulty
with torch.no_grad():
    rec_healthy = model(X_train[0:1]) # Test one healthy sample
    rec_faulty = model(X_test_faulty[0:1]) # Test one faulty sample
    
    # Calculate Reconstruction Error (Anomaly Score)
    loss_healthy = criterion(rec_healthy, X_train[0:1]).item()
    loss_faulty = criterion(rec_faulty, X_test_faulty[0:1]).item()

print(f"Healthy Reconstruction Error: {loss_healthy:.6f}")
print(f"Faulty Reconstruction Error:  {loss_faulty:.6f} (Anomaly Detected!)")

# Plotting the "Money Shot"
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Loss
axs[0].plot(losses, color='blue')
axs[0].set_title('Model Training (Loss minimization)')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('MSE Loss')

# Plot 2: Healthy Reconstruction
axs[1].plot(X_train[0].numpy(), label='Original (Sensor)', color='green')
axs[1].plot(rec_healthy[0].numpy(), label='AI Reconstructed', linestyle='--', color='black')
axs[1].set_title(f'Healthy Cycle Validation\nError: {loss_healthy:.4f}')
axs[1].legend()

# Plot 3: Faulty Reconstruction
axs[2].plot(X_test_faulty[0].numpy(), label='Original (Sensor)', color='red')
axs[2].plot(rec_faulty[0].numpy(), label='AI Reconstructed', linestyle='--', color='black')
axs[2].set_title(f'Anomaly Detection (Aging)\nError: {loss_faulty:.4f}')
axs[2].legend()

plt.tight_layout()
plt.savefig('results_dashboard.png')
print("✅ Done! Results saved to results_dashboard.png")
plt.show()