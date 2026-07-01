# Intrusion Detection System (IDS) - Packet Monitoring

An ML-based intrusion detection system that monitors live network traffic and detects anomalies using a TensorFlow autoencoder neural network.

## Features

- **Live packet capture** using Scapy
- **Anomaly detection** with TensorFlow/Keras autoencoder
- **Real-time display** with colored terminal output
- **Detailed packet inspection** including source/destination, protocols, flags, and payload previews
- **Training mode** for collecting baseline "normal" traffic

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Mode (First Run - Collect Normal Traffic)

Run in training mode to collect baseline data for "normal" network traffic:

```bash
python main.py --train
```

Let it capture ~1000 packets of normal traffic, then press Ctrl+C to save the trained model.

### Detection Mode

After training, run in detection mode to monitor for anomalies:

```bash
python main.py
```

### Options

```bash
python main.py --help
```

- `--train` - Training mode to collect normal traffic data
- `--interface INTERFACE` - Specify network interface (e.g., `eth0`, `Wi-Fi`)
- `--threshold FLOAT` - Anomaly detection threshold (default: 0.7)

## How It Works

1. **Packet Capture**: Uses Scapy to capture live network packets
2. **Feature Extraction**: Extracts numerical features (length, TTL, protocol, ports, flags, payload size, time)
3. **Preprocessing**: Normalizes features with StandardScaler and reduces dimensionality with PCA
4. **Autoencoder**: Neural network learns to reconstruct "normal" traffic patterns
5. **Anomaly Detection**: High reconstruction error = potential intrusion

## Detected Anomalies Include

- Unusual packet sizes
- Suspicious port activity
- Abnormal flag combinations
- Irregular traffic patterns
- Potential scanning/probing activity

## Requirements

- Python 3.8+
- Administrator/root privileges for packet capture
- Windows: Npcap installed (or run as Administrator)
- Linux: Root privileges or setcap for Scapy

## Files Generated

- `ids_model.keras` - Trained TensorFlow autoencoder model
- `scaler.npz` - Saved preprocessing parameters

## Security Note

This IDS is designed for **authorized security monitoring only**:
- Use only on networks you own or have permission to monitor
- Do not use for unauthorized surveillance or network reconnaissance