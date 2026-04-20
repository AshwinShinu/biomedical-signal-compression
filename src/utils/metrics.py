
import torch
import numpy as np

def calculate_prd(original, reconstructed):
    """
    Percent Root Difference (PRD).
    Lower is better. Works on both raw and Z-scored signals.
    """
    diff = original - reconstructed
    num = torch.sum(diff ** 2)
    den = torch.sum(original ** 2)
    
    return torch.sqrt(num / (den + 1e-8)) * 100

def calculate_prd_batch(original, reconstructed):
    """
    Energy-weighted PRD over a batch.
    
    Instead of averaging per-window PRD (which inflates due to low-energy windows),
    this accumulates numerator and denominator across the full batch before dividing.
    
    This gives the TRUE reconstruction error relative to total signal energy.
    """
    diff = original - reconstructed
    num = torch.sum(diff ** 2)
    den = torch.sum(original ** 2)
    return torch.sqrt(num / (den + 1e-8)) * 100

def calculate_snr(original, reconstructed):
    """
    Signal-to-Noise Ratio (dB).
    Higher is better.
    """
    noise = original - reconstructed
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    
    return 10 * torch.log10(signal_power / (noise_power + 1e-8))

def cosine_similarity(v1, v2):
    return torch.nn.functional.cosine_similarity(v1, v2, dim=-1).mean()

def calculate_cr(original_size, compressed_size):
    """
    Calculate Compression Ratio.
    CR = Original Size / Compressed Size
    """
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size
