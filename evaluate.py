"""
Standalone Evaluation Script
Loads a trained VQ-VAE checkpoint, runs the test set, and prints a clean results table.
Usage: python evaluate.py [--checkpoint checkpoints/vq_model_best.pth]
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vq_vae import DualVQVAE
from data.paired_dataset import PairedEEGECGDataset
from utils.metrics import calculate_prd, calculate_snr, calculate_cr


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # Load model
    model = DualVQVAE(num_embeddings=1024, hidden_dim=256).to(device)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Load test data
    data_dir = os.path.join(args.data_dir, 'cap-sleep')
    test_dataset = PairedEEGECGDataset(data_dir, split='test', test_subjects=['n9', 'n10'])
    
    if len(test_dataset) == 0:
        print("ERROR: No test data found.")
        return

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # --- Accumulate metrics across entire test set ---
    eeg_diff_sq, eeg_orig_sq = 0.0, 0.0
    ecg_diff_sq, ecg_orig_sq = 0.0, 0.0
    eeg_snr_list, ecg_snr_list = [], []
    n_samples = 0

    print("Running evaluation...\n")
    with torch.no_grad():
        for eeg_batch, ecg_batch in test_loader:
            eeg_batch = eeg_batch.to(device)
            ecg_batch = ecg_batch.to(device)
            
            rec_eeg, rec_ecg, vq_loss_eeg, vq_loss_ecg = model(eeg_batch, ecg_batch)

            # Global PRD accumulation
            eeg_diff_sq += torch.sum((eeg_batch - rec_eeg) ** 2).item()
            eeg_orig_sq += torch.sum(eeg_batch ** 2).item()
            ecg_diff_sq += torch.sum((ecg_batch - rec_ecg) ** 2).item()
            ecg_orig_sq += torch.sum(ecg_batch ** 2).item()

            # Per-batch SNR
            eeg_snr_list.append(calculate_snr(eeg_batch, rec_eeg).item())
            ecg_snr_list.append(calculate_snr(ecg_batch, rec_ecg).item())

            n_samples += eeg_batch.size(0)

    # --- Compute final metrics ---
    prd_eeg = (eeg_diff_sq / (eeg_orig_sq + 1e-8)) ** 0.5 * 100
    prd_ecg = (ecg_diff_sq / (ecg_orig_sq + 1e-8)) ** 0.5 * 100
    snr_eeg = np.mean(eeg_snr_list)
    snr_ecg = np.mean(ecg_snr_list)

    # Compression ratio: 256 samples * 32-bit input → 32 codebook IDs * 10-bit
    cr_dimensional = calculate_cr(256 * 32, 32 * 10)  # 8:1
    cr_with_int8 = calculate_cr(256 * 32, 32 * 8)     # 16:1 (with INT8 quantization)

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # --- Print Results Table ---
    print("=" * 60)
    print("         EVALUATION RESULTS — Dual VQ-VAE")
    print("=" * 60)
    print(f"  Test Samples:           {n_samples}")
    print(f"  Model Parameters:       {param_count:,}")
    print(f"  Model Size (FP32):      {model_size_mb:.2f} MB")
    print("-" * 60)
    print(f"  {'Metric':<30} {'ECG':>10} {'EEG':>10}")
    print("-" * 60)
    print(f"  {'PRD (%)':<30} {prd_ecg:>10.2f} {prd_eeg:>10.2f}")
    print(f"  {'SNR (dB)':<30} {snr_ecg:>10.2f} {snr_eeg:>10.2f}")
    print(f"  {'Compression Ratio (dim)':<30} {cr_dimensional:>10.1f}x {cr_dimensional:>10.1f}x")
    print(f"  {'Compression Ratio (+INT8)':<30} {cr_with_int8:>10.1f}x {cr_with_int8:>10.1f}x")
    print("=" * 60)

    # Quality classification (Zigel et al., 2000)
    print("\n  ECG PRD Quality Classification (Zigel et al.):")
    if prd_ecg < 2:
        print("    → \"Very Good\" (PRD < 2%)")
    elif prd_ecg < 9:
        print("    → \"Good\" — suitable for diagnostic use (PRD < 9%)")
    elif prd_ecg < 20:
        print("    → \"Acceptable\" — suitable for monitoring (PRD < 20%)")
    else:
        print(f"    → \"Needs improvement\" (PRD = {prd_ecg:.1f}%)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained VQ-VAE model")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vq_model_best.pth')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
