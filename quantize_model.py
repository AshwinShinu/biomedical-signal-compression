"""
INT8 Quantization Pipeline
Quantizes the trained VQ-VAE model and compares size + inference speed.
Usage: python quantize_model.py [--checkpoint checkpoints/vq_model_best.pth]
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vq_vae import DualVQVAE
from data.paired_dataset import PairedEEGECGDataset
from utils.metrics import calculate_prd


def get_model_size(model):
    """Calculate model size in bytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size


def benchmark_inference(model, eeg_input, ecg_input, n_runs=100):
    """Time inference over multiple runs."""
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(eeg_input, ecg_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(eeg_input, ecg_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def quantize(args):
    device = torch.device('cpu')  # Quantization works on CPU
    print("INT8 Quantization Pipeline")
    print("=" * 50)

    # Load original model
    model_fp32 = DualVQVAE(num_embeddings=1024, hidden_dim=256).to(device)
    model_fp32.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model_fp32.eval()

    size_fp32 = get_model_size(model_fp32)
    print(f"\nOriginal FP32 model size: {size_fp32 / (1024*1024):.2f} MB")

    # Dynamic INT8 quantization (quantizes weights of linear/conv layers)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Conv1d, nn.ConvTranspose1d, nn.Linear},
        dtype=torch.qint8
    )

    size_int8 = get_model_size(model_int8)
    print(f"Quantized INT8 model size: {size_int8 / (1024*1024):.2f} MB")
    print(f"Size reduction: {size_fp32 / size_int8:.2f}x")

    # Save quantized model
    os.makedirs('checkpoints', exist_ok=True)
    output_path = 'checkpoints/vq_model_int8.pth'
    torch.save(model_int8.state_dict(), output_path)
    
    # Also save as full model for size measurement
    scripted_path = 'checkpoints/vq_model_int8_full.pt'
    try:
        torch.save(model_int8, scripted_path)
        actual_file_size = os.path.getsize(scripted_path) / (1024*1024)
        print(f"Saved file size on disk: {actual_file_size:.2f} MB")
    except Exception as e:
        print(f"Note: Full model save skipped ({e})")

    # --- Compare accuracy ---
    print("\n" + "=" * 50)
    print("Accuracy Comparison (FP32 vs INT8)")
    print("=" * 50)

    data_dir = os.path.join(args.data_dir, 'cap-sleep')
    if os.path.exists(data_dir):
        dataset = PairedEEGECGDataset(data_dir, split='test', test_subjects=['n9', 'n10'])
        if len(dataset) > 0:
            # Test on a subset
            n_test = min(50, len(dataset))
            
            fp32_ecg_prds, fp32_eeg_prds = [], []
            int8_ecg_prds, int8_eeg_prds = [], []

            with torch.no_grad():
                for i in range(n_test):
                    eeg, ecg = dataset[i]
                    eeg_in = eeg.unsqueeze(0).to(device)
                    ecg_in = ecg.unsqueeze(0).to(device)

                    # FP32
                    rec_eeg_fp32, rec_ecg_fp32, _, _ = model_fp32(eeg_in, ecg_in)
                    fp32_ecg_prds.append(calculate_prd(ecg_in, rec_ecg_fp32).item())
                    fp32_eeg_prds.append(calculate_prd(eeg_in, rec_eeg_fp32).item())

                    # INT8
                    rec_eeg_int8, rec_ecg_int8, _, _ = model_int8(eeg_in, ecg_in)
                    int8_ecg_prds.append(calculate_prd(ecg_in, rec_ecg_int8).item())
                    int8_eeg_prds.append(calculate_prd(eeg_in, rec_eeg_int8).item())

            print(f"\n  {'Metric':<25} {'FP32':>10} {'INT8':>10} {'Δ':>10}")
            print("-" * 60)
            print(f"  {'ECG PRD (%) ↓':<25} {np.mean(fp32_ecg_prds):>10.2f} {np.mean(int8_ecg_prds):>10.2f} {np.mean(int8_ecg_prds)-np.mean(fp32_ecg_prds):>+10.2f}")
            print(f"  {'EEG PRD (%) ↓':<25} {np.mean(fp32_eeg_prds):>10.2f} {np.mean(int8_eeg_prds):>10.2f} {np.mean(int8_eeg_prds)-np.mean(fp32_eeg_prds):>+10.2f}")

    # --- Benchmark inference speed ---
    print("\n" + "=" * 50)
    print("Inference Speed (CPU)")
    print("=" * 50)

    dummy_eeg = torch.randn(1, 1, 256)
    dummy_ecg = torch.randn(1, 1, 256)

    fp32_time, fp32_std = benchmark_inference(model_fp32, dummy_eeg, dummy_ecg)
    int8_time, int8_std = benchmark_inference(model_int8, dummy_eeg, dummy_ecg)

    print(f"  FP32: {fp32_time:.2f} ± {fp32_std:.2f} ms")
    print(f"  INT8: {int8_time:.2f} ± {int8_std:.2f} ms")
    print(f"  Speedup: {fp32_time/int8_time:.2f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize VQ-VAE model to INT8")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vq_model_best.pth')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    quantize(args)
