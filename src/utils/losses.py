
import torch
import torch.nn as nn
import torch.fft

class PRDLoss(nn.Module):
    """
    Differentiable PRD loss — directly optimizes what we measure.
    PRD = sqrt(sum((x - x_hat)^2) / sum(x^2)) 
    """
    def forward(self, pred, target):
        diff = target - pred
        num = torch.sum(diff ** 2)
        den = torch.sum(target ** 2) + 1e-8
        return torch.sqrt(num / den)

class SpectralLoss(nn.Module):
    """Lightweight spectral loss — supporting role."""
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)))

class MedicalFeatureLoss(nn.Module):
    """
    Combined loss with modality-specific weighting.
    ECG: Needs morphological accuracy (MSE + PRD).
    EEG: Needs frequency accuracy (Spectral) because time-domain stochastic matching fails.
    """
    def __init__(self):
        super(MedicalFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.prd = PRDLoss()
        self.spectral = SpectralLoss()
    
    def forward(self, pred, target, signal_type='ecg'):
        l_mse = self.mse(pred, target)
        l_prd = self.prd(pred, target)
        l_spec = self.spectral(pred, target)
        
        if signal_type == 'ecg':
            # ECG relies on shape and energy -> 40% MSE, 40% PRD, 20% Spectral
            return 0.4 * l_mse + 0.4 * l_prd + 0.2 * l_spec
        else:
            # EEG relies on frequency bands -> 60% Spectral, 20% PRD, 20% MSE
            # Lowering PRD/MSE prevents the network from panicking over stochastic phase differences
            return 0.2 * l_mse + 0.2 * l_prd + 0.6 * l_spec
