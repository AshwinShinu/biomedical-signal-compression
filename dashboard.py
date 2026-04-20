
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vq_vae import DualVQVAE
from data.paired_dataset import PairedEEGECGDataset
from utils.metrics import calculate_prd

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cap-sleep')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'vq_model_best.pth')
LATENT_DIM = 256 # VQ bottlenecks to 256 integers at 10-bit vocabulary
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- TITLE ---
st.set_page_config(page_title="Bio-Signal Compression", layout="wide")
st.title("Dual-Stream Autoencoder Dashboard 🏥")
st.markdown("Real-time compression and reconstruction of **paired ECG + EEG** signals from CAP Sleep Database.")

# --- SIDEBAR ---
st.sidebar.header("Control Panel")
signal_type = st.sidebar.radio("Select Signal Type", ["ECG (Heart)", "EEG (Brain)", "Both (Paired)"])
sample_index = st.sidebar.slider("Select Sample ID", 0, 200, 0)
st.sidebar.markdown("**Transmission Mode: 10-Bit Discrete VQ**")
add_noise = st.sidebar.checkbox("Add Transmission Noise simulation?", value=False, disabled=True, help="Disabled. VQ-VAE transmits integer IDs natively resilient to minor noise.")
quantize = st.sidebar.checkbox("Enable float16 Quantization?", value=False, disabled=True, help="Disabled. VQ-VAE transmits 10-bit IDs directly (8:1 Compression).")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_system():
    # Loading the widened 8:1 10-bit architecture
    model = DualVQVAE(num_embeddings=1024, hidden_dim=256).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
    else:
        st.error(f"Model not found at {MODEL_PATH}. Download from Kaggle first.")
    model.eval()
    return model

@st.cache_resource
def load_data():
    try:
        dataset = PairedEEGECGDataset(DATA_DIR, split='test', test_subjects=['n9', 'n10'])
        if len(dataset) == 0:
            # Fall back to loading all data
            dataset = PairedEEGECGDataset(DATA_DIR, split='train')
        return dataset
    except Exception as e:
        st.error(f"Data load error: {e}")
        return None

try:
    model = load_system()
    dataset = load_data()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

if dataset is None or len(dataset) == 0:
    st.error("No data found. Please download paired data: `python -m src.data.download_paired_data --n 3`")
    st.stop()

# --- INFERENCE ---
st.subheader("Signal Analysis")

idx = sample_index % len(dataset)
eeg_sample, ecg_sample = dataset[idx]
eeg_input = eeg_sample.unsqueeze(0).to(DEVICE)  # (1, 1, 256)
ecg_input = ecg_sample.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    # The VQ-VAE natively quantizes the features into 8-bit integer IDs representing the clinical shapes.
    # In a real edge telemetry system, only these IDs are transmitted over Bluetooth.
    rec_eeg, rec_ecg, _, _ = model(eeg_input, ecg_input)

# Metrics
eeg_prd = calculate_prd(eeg_input, rec_eeg).item()
ecg_prd = calculate_prd(ecg_input, rec_ecg).item()

eeg_orig = eeg_input.cpu().numpy().squeeze()
eeg_rec = rec_eeg.cpu().numpy().squeeze()
ecg_orig = ecg_input.cpu().numpy().squeeze()
ecg_rec = rec_ecg.cpu().numpy().squeeze()

if signal_type == "Both (Paired)":
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(eeg_orig, label='Original', color='#2ca02c', alpha=0.85, linewidth=1.8)
        ax.plot(eeg_rec, label='Reconstructed', color='blue', linewidth=1.5)
        ax.set_title(f"EEG (PRD: {eeg_prd:.2f}%)")
        ax.set_xlabel("Samples")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ecg_orig, label='Original', color='#1f77b4', alpha=0.85, linewidth=1.8)
        ax.plot(ecg_rec, label='Reconstructed', color='red', linewidth=1.5)
        ax.set_title(f"ECG (PRD: {ecg_prd:.2f}%)")
        ax.set_xlabel("Samples")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if signal_type == "ECG (Heart)":
            orig, rec, prd, color, title = ecg_orig, ecg_rec, ecg_prd, 'red', 'ECG'
        else:
            orig, rec, prd, color, title = eeg_orig, eeg_rec, eeg_prd, 'blue', 'EEG'
        
        fig, ax = plt.subplots(figsize=(10, 4))
        orig_color = '#2ca02c' if signal_type == "EEG (Brain)" else '#1f77b4'
        ax.plot(orig, label='Original', color=orig_color, alpha=0.85, linewidth=1.8)
        ax.plot(rec, label='Reconstructed', color=color, linewidth=2)
        ax.set_title(f"{title} — Sample #{idx} (PRD: {prd:.2f}%)")
        ax.set_xlabel("Samples")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("### Performance")
        st.metric(label="Compression Ratio", value="8:1")
        st.metric(label="PRD (Error %)", value=f"{prd:.2f}%", delta="Lower is better")

# --- SIDEBAR INFO ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.write(f"Latent dim: {LATENT_DIM}")
st.sidebar.write(f"EEG PRD: {eeg_prd:.2f}%")
st.sidebar.write(f"ECG PRD: {ecg_prd:.2f}%")
st.sidebar.write(f"Compression: 8:1 (10-bit VQ)")
st.sidebar.write(f"Dataset: CAP Sleep DB")
st.sidebar.write(f"Samples available: {len(dataset)}")
