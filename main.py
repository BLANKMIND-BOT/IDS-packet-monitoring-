#!/usr/bin/env python3
"""
Intrusion Detection System (IDS) with Machine Learning
Monitors live network traffic and detects anomalies using TensorFlow autoencoder.
"""

import os
import sys
import time
import json
import threading
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from tabulate import tabulate
from colorama import init, Fore, Style
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
from scapy.layers.inet import Ether
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configuration
CONFIG = {
    'model_path': 'ids_model.keras',
    'scaler_path': 'scaler.npz',
    'features_path': 'training_features.json',
    'interface': None,  # Auto-detect interface
    'packet_buffer_size': 1000,
    'training_mode': False,  # Set True to collect training data
    'anomaly_threshold': 0.7,  # Reconstruction error threshold
    'display_interval': 2,  # Seconds between display updates
}


class PacketFeatureExtractor:
    """Extract numerical features from network packets for ML analysis."""

    def __init__(self):
        self.feature_names = [
            'packet_len', 'ip_ttl', 'ip_proto', 'src_port', 'dst_port',
            'tcp_flags_value', 'payload_len', 'is_syn', 'is_ack', 'is_fin',
            'is_rst', 'is_psh', 'is_urg', 'hour', 'minute'
        ]

    def extract(self, packet) -> list:
        """Extract features from a single packet."""
        features = np.zeros(len(self.feature_names))

        # Basic packet length
        features[0] = len(packet)

        if IP not in packet:
            return features

        # IP layer features
        ip = packet[IP]
        features[1] = ip.ttl
        features[2] = ip.proto

        # Transport layer features
        if TCP in packet:
            tcp = packet[TCP]
            features[3] = tcp.sport
            features[4] = tcp.dport
            flags = tcp.flags
            features[5] = int(flags.value)
            features[6] = len(packet[Raw].load) if Raw in packet else 0
            features[7] = 1 if flags & 'S' else 0  # SYN
            features[8] = 1 if flags & 'A' else 0  # ACK
            features[9] = 1 if flags & 'F' else 0  # FIN
            features[10] = 1 if flags & 'R' else 0  # RST
            features[11] = 1 if flags & 'P' else 0  # PSH
            features[12] = 1 if flags & 'U' else 0  # URG
        elif UDP in packet:
            udp = packet[UDP]
            features[3] = udp.sport
            features[4] = udp.dport
            features[6] = len(packet[Raw].load) if Raw in packet else 0
        elif ICMP in packet:
            features[2] = 1  # ICMP protocol
            features[6] = len(packet[Raw].load) if Raw in packet else 0

        # Payload length (bytes)
        if Raw in packet:
            features[6] = len(packet[Raw].load)

        # Time-based features (indices 13, 14)
        now = datetime.now()
        features[13] = now.hour
        features[14] = now.minute

        return features


class IDSModel:
    """TensorFlow autoencoder for anomaly-based intrusion detection."""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(10, input_dim))
        self.threshold = CONFIG['anomaly_threshold']
        self._build_model()

    def _build_model(self):
        """Build the autoencoder model architecture."""
        tf.random.set_seed(42)

        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(32, activation='relu')(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(16, activation='relu')(encoded)

        # Decoder
        decoded = keras.layers.Dense(32, activation='relu')(encoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        decoded = keras.layers.Dense(64, activation='relu')(decoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        output_layer = keras.layers.Dense(self.input_dim, activation='linear')(decoded)

        self.model = keras.Model(input_layer, output_layer)
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the autoencoder on normal traffic data."""
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        # Train autoencoder
        self.model.fit(
            X_pca, X_pca,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            verbose=0
        )

        # Calculate threshold from reconstruction errors
        reconstructed = self.model.predict(X_pca, verbose=0)
        errors = np.mean(np.square(X_pca - reconstructed), axis=1)
        self.threshold = np.percentile(errors, 95)  # 95th percentile as threshold

    def predict(self, X: np.ndarray) -> tuple:
        """Predict if packets are anomalous. Returns (is_anomaly, reconstruction_error)."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        reconstructed = self.model.predict(X_pca, verbose=0)
        errors = np.mean(np.square(X_pca - reconstructed), axis=1)

        is_anomaly = errors > self.threshold
        return is_anomaly, errors

    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler to disk."""
        self.model.save(model_path)
        np.savez(scaler_path, scaler_mean=self.scaler.mean_, scaler_scale=self.scaler.scale_,
                 pca_components=self.pca.components_, pca_mean=self.pca.mean_)

    def load(self, model_path: str, scaler_path: str):
        """Load model and scaler from disk."""
        self.model = keras.models.load_model(model_path)
        data = np.load(scaler_path, allow_pickle=True)
        self.scaler.mean_ = data['scaler_mean']
        self.scaler.scale_ = data['scaler_scale']
        self.pca.components_ = data['pca_components']
        self.pca.mean_ = data['pca_mean']


class IntrusionDetector:
    """Main IDS class that coordinates packet capture and detection."""

    def __init__(self):
        self.feature_extractor = PacketFeatureExtractor()
        self.packet_buffer = deque(maxlen=CONFIG['packet_buffer_size'])
        self.packet_features = []
        self.model = IDSModel(input_dim=len(self.feature_extractor.feature_names))
        self.running = False
        self.stats = {
            'total_packets': 0,
            'alerts': 0,
            'normal': 0
        }
        self.recent_alerts = deque(maxlen=20)

    def packet_handler(self, packet):
        """Process each captured packet."""
        self.stats['total_packets'] += 1

        # Extract features
        features = self.feature_extractor.extract(packet)
        features_list = features.tolist()

        # Store for training
        if CONFIG['training_mode']:
            self.packet_features.append(features_list)
            # Show progress every 100 packets
            if len(self.packet_features) % 100 == 0:
                print(f"Training progress: {len(self.packet_features)} packets collected...")
            # Alert when we have enough training data
            if len(self.packet_features) >= CONFIG['packet_buffer_size']:
                print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ Collected {CONFIG['packet_buffer_size']} packets.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Press Ctrl+C to stop training and save the model.{Style.RESET_ALL}")
            return

        # Detect anomalies
        features_array = np.array([features_list])
        is_anomaly, error = self.model.predict(features_array)

        # Create packet info dict
        packet_info = {
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'src': self._get_src(packet),
            'dst': self._get_dst(packet),
            'proto': self._get_proto(packet),
            'length': len(packet),
            'flags': self._get_flags(packet),
            'payload_preview': self._get_payload_preview(packet),
            'anomaly_score': float(error[0]),
            'is_alert': bool(is_anomaly[0])
        }

        # Add to display buffer
        self.packet_buffer.append(packet_info)

        if packet_info['is_alert']:
            self.stats['alerts'] += 1
            self.recent_alerts.appendleft(packet_info)
            self._display_alert(packet_info)
        else:
            self.stats['normal'] += 1

    def _get_src(self, packet) -> str:
        """Get source IP:port from packet."""
        if IP in packet:
            src = f"{packet[IP].src}"
            if TCP in packet:
                src += f":{packet[TCP].sport}"
            elif UDP in packet:
                src += f":{packet[UDP].sport}"
        else:
            src = "N/A"
        return src

    def _get_dst(self, packet) -> str:
        """Get destination IP:port from packet."""
        if IP in packet:
            dst = f"{packet[IP].dst}"
            if TCP in packet:
                dst += f":{packet[TCP].dport}"
            elif UDP in packet:
                dst += f":{packet[UDP].dport}"
        else:
            dst = "N/A"
        return dst

    def _get_proto(self, packet) -> str:
        """Get protocol name from packet."""
        if TCP in packet:
            return "TCP"
        elif UDP in packet:
            return "UDP"
        elif ICMP in packet:
            return "ICMP"
        elif IP in packet:
            return f"IP({packet[IP].proto})"
        else:
            return "Other"

    def _get_flags(self, packet) -> str:
        """Get TCP flags from packet."""
        if TCP in packet:
            flags = packet[TCP].flags
            return flags if flags else "-"
        return "-"

    def _get_payload_preview(self, packet) -> str:
        """Get a preview of the payload data."""
        if Raw in packet:
            try:
                payload = packet[Raw].load[:50]
                # Try to decode as text, fallback to hex
                try:
                    return payload.decode('utf-8', errors='replace')
                except:
                    return payload.hex()
            except:
                return ""
        return ""

    def _display_alert(self, packet_info: dict):
        """Display an alert for anomalous traffic."""
        print(f"\n{Fore.RED}{Style.BRIGHT}🚨 ALERT: Anomalous Traffic Detected!{Style.RESET_ALL}")
        print(f"  Time: {packet_info['timestamp']}")
        print(f"  Source: {packet_info['src']}")
        print(f"  Destination: {packet_info['dst']}")
        print(f"  Protocol: {packet_info['proto']}")
        print(f"  Length: {packet_info['length']} bytes")
        print(f"  Anomaly Score: {packet_info['anomaly_score']:.4f}")
        if packet_info['payload_preview']:
            print(f"  Payload: {packet_info['payload_preview'][:100]}")
        print()

    def display_status(self):
        """Periodically display packet statistics and recent activity."""
        while self.running:
            time.sleep(CONFIG['display_interval'])
            self._clear_screen()
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}INTRUSION DETECTION SYSTEM - LIVE MONITOR{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

            # Display statistics
            print(f"\n{Fore.GREEN}Statistics:{Style.RESET_ALL}")
            stats_table = [
                ['Total Packets', self.stats['total_packets']],
                ['Normal Traffic', self.stats['normal']],
                ['Alerts', self.stats['alerts']],
                ['Alert Rate', f"{self.stats['alerts']/max(1,self.stats['total_packets'])*100:.2f}%"]
            ]
            print(tabulate(stats_table, tablefmt='simple'))

            # Display recent packets
            if self.packet_buffer:
                print(f"\n{Fore.YELLOW}Recent Packets (Last {min(10, len(self.packet_buffer))}):{Style.RESET_ALL}")
                display_packets = list(self.packet_buffer)[-10:]
                packet_table = []
                for p in display_packets:
                    color = Fore.RED if p.get('is_alert') else Fore.GREEN
                    packet_table.append([
                        p.get('timestamp', '-'),
                        p.get('src', '-'),
                        p.get('dst', '-'),
                        p.get('proto', '-'),
                        p.get('length', 0),
                        f"{color}{'ALERT' if p.get('is_alert') else 'OK'}{Style.RESET_ALL}"
                    ])

                headers = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Status']
                print(tabulate(packet_table, headers=headers, tablefmt='grid'))

            # Display recent alerts
            if self.recent_alerts:
                print(f"\n{Fore.RED}Recent Alerts:{Style.RESET_ALL}")
                for alert in list(self.recent_alerts)[:5]:
                    print(f"  [{alert['timestamp']}] {alert['src']} → {alert['dst']} "
                          f"(Score: {alert['anomaly_score']:.4f})")

    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def start(self):
        """Start the IDS packet capture."""
        print(f"{Fore.CYAN}Starting IDS packet capture...{Style.RESET_ALL}")

        if CONFIG['training_mode']:
            print(f"{Fore.YELLOW}Training mode active - collecting normal traffic data...{Style.RESET_ALL}")
            print(f"Press Ctrl+C to stop and save training data.")
        else:
            # Check if model exists, train if not
            if os.path.exists(CONFIG['model_path']):
                print(f"{Fore.GREEN}Loading existing model...{Style.RESET_ALL}")
                self.model.load(CONFIG['model_path'], CONFIG['scaler_path'])
            else:
                print(f"{Fore.YELLOW}No model found. Training on first {CONFIG['packet_buffer_size']} packets.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Assuming initial traffic is normal...{Style.RESET_ALL}")

        self.running = True

        # Start display thread
        if not CONFIG['training_mode']:
            display_thread = threading.Thread(target=self.display_status)
            display_thread.daemon = True
            display_thread.start()

        try:
            sniff(
                prn=self.packet_handler,
                store=0,
                iface=CONFIG['interface']
            )
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping IDS...{Style.RESET_ALL}")

        self.running = False

        if CONFIG['training_mode'] and self.packet_features:
            self._save_training_data()

    def _save_training_data(self):
        """Save collected training data."""
        features = np.array(self.packet_features)
        self.model.train(features)
        self.model.save(CONFIG['model_path'], CONFIG['scaler_path'])
        print(f"{Fore.GREEN}Model trained and saved!{Style.RESET_ALL}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='ML-Based Intrusion Detection System')
    parser.add_argument('--train', action='store_true',
                        help='Training mode - collect normal traffic for model training')
    parser.add_argument('--interface', type=str, default=None,
                        help='Network interface to capture on (default: auto)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Anomaly detection threshold (default: 0.7)')

    args = parser.parse_args()

    CONFIG['training_mode'] = args.train
    CONFIG['interface'] = args.interface
    CONFIG['anomaly_threshold'] = args.threshold

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   INTRUSION DETECTION SYSTEM (IDS) v1.0{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    detector = IntrusionDetector()
    detector.start()


if __name__ == '__main__':
    main()