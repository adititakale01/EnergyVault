# EnergyVault: Deep Learning for Battery Anomaly Detection 🔋

> **Status:** Prototype  
> **Tech Stack:** Python, PyTorch, LSTM, Matplotlib  
> **Context:** Validation of Energy Systems / Test Bench Data Analysis

## Project Overview
**EnergyVault** is a data-driven validation pipeline designed to detect anomalies in Li-Ion battery charging cycles. By implementing an **LSTM Autoencoder** (Neural Network), the system learns the "fingerprint" of healthy battery performance and autonomously flags deviations caused by aging, sensor drift, or thermal inefficiencies.

This project simulates the workflow of analyzing **high-frequency test bench data (Prüfstandsdaten)**, bridging the gap between physical validation and advanced Machine Learning methods.

## 🛠️ Key Features
*   **Synthetic Data Generation:** Simulates realistic voltage curves for healthy vs. degraded batteries (replicating test bench sensor streams).
*   **LSTM Autoencoder:** A time-series Neural Network implemented in **PyTorch** for unsupervised anomaly detection.
*   **Automated Validation:** Calculates reconstruction error metrics to distinguish between "Safe" and "Critical" states.
*   **Visualization:** Automated dashboard generation for model performance review.

## 📊 Results
The model successfully identifies aging batteries by analyzing the reconstruction error (MSE).

![Results Dashboard](results_dashboard.png)

*   **Left:** Training convergence (Model learning physics).
*   **Center:** Healthy battery (Low error, perfect reconstruction).
*   **Right:** **Anomaly Detected** (High error due to voltage drift/aging).

## 💻 How to Run

1. **Clone the repository.**
   ```bash
   git clone https://github.com/adititakale01/EnergyVault.git
   cd EnergyVault
   ```

2. **Install dependencies.**
   
   *Option A: Standard Install (if you have good internet)*
   ```bash
   pip install -r requirements.txt
   ```

   *Option B: Fast Install (CPU Only - Recommended)* 🚀
   *Run these two commands separately to avoid conflicts:*
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install numpy matplotlib pandas scikit-learn
   ```

3. **Run the prototyping script.**
   ```bash
   python main.py
   ```
   *This will train the model, validate it, and generate the `results_dashboard.png` image.*

## Why this matters
In modern automotive R&D, validating energy systems requires more than static thresholds. This project demonstrates a **"Prototyping Mindset"** by applying **Deep Learning** to time-series data, automating the detection of subtle errors that traditional methods often miss.
