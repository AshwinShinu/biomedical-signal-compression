import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    Codebook with Exponential Moving Average (EMA) updates.
    This prevents 'Codebook Collapse' (where the model only uses 1 or 2 shapes).
    """
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_embedding', torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self.register_buffer('_ema_w', torch.randn(self._num_embeddings, self._embedding_dim))
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # Force float32 — distance computation overflows in float16
        input_dtype = inputs.dtype
        inputs = inputs.float()
        
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size.data.mul_(self._decay).add_(
                    torch.sum(encodings, 0), alpha=(1 - self._decay)
                )
                
                # Laplace smoothing (prevents divide by zero)
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size.data = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n
                )
                
                dw = torch.matmul(encodings.t(), flat_input.float())
                self._ema_w.data.mul_(self._decay).add_(dw, alpha=(1 - self._decay))
                
                self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
                
                # Restart Dead Codes (if a shape isn't used, replace it with a random input shape)
                dead_codes = self._ema_cluster_size < 1.0
                if dead_codes.any():
                    random_indices = torch.randint(0, flat_input.size(0), (dead_codes.sum(),))
                    self._embedding.data[dead_codes] = flat_input[random_indices].data.float()
                    self._ema_w.data[dead_codes] = flat_input[random_indices].data.float()
                    self._ema_cluster_size.data[dead_codes] = 1.0

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Cast back to input dtype for AMP compatibility
        quantized = quantized.to(input_dtype)
        
        return quantized, loss, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + residual, 0.1)

class VQEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super(VQEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2), # 128
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            ResidualBlock(hidden_dim),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1), # 64
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            ResidualBlock(hidden_dim * 2),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1), # 32
        )

    def forward(self, x):
        return self.net(x)

class VQDecoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(VQDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1), # 64
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            ResidualBlock(hidden_dim * 2),
            
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1), # 128
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            ResidualBlock(hidden_dim),
            
            nn.ConvTranspose1d(hidden_dim, 1, kernel_size=5, stride=2, padding=2, output_padding=1), # 256
        )

    def forward(self, x):
        return self.net(x)

class DualVQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, hidden_dim=256):
        super(DualVQVAE, self).__init__()
        # Two independent encoders for physiological specificity
        self.eeg_encoder = VQEncoder(1, hidden_dim)
        self.ecg_encoder = VQEncoder(1, hidden_dim)
        
        # Pre-Quantization Projection (Smooths the encoder outputs to match codebook geometry)
        self.eeg_pre_vq = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        self.ecg_pre_vq = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        
        # The Semantic Codebooks (STABILIZED WITH EMA)
        self.eeg_vq = VectorQuantizerEMA(num_embeddings, embedding_dim=hidden_dim * 2)
        self.ecg_vq = VectorQuantizerEMA(num_embeddings, embedding_dim=hidden_dim * 2)
        
        # Post-Quantization Projection
        self.eeg_post_vq = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        self.ecg_post_vq = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1)
        
        # Cross-Modal Attention (Allows EEG to query ECG states for context)
        # Using 8 heads to maintain dimensionality balance
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8, batch_first=True)
        
        # Two independent decoders
        self.eeg_decoder = VQDecoder(hidden_dim)
        self.ecg_decoder = VQDecoder(hidden_dim)

    def forward(self, eeg, ecg):
        # 1. Encode
        z_e = self.eeg_encoder(eeg) # z_e: (B, 256, 32)
        z_c = self.ecg_encoder(ecg)
        
        # Pre-VQ Projection
        z_e = self.eeg_pre_vq(z_e)
        z_c = self.ecg_pre_vq(z_c)
        
        # Reshape for VQ (B, C, L) -> (B, L, C)
        z_e = z_e.permute(0, 2, 1).contiguous()
        z_c = z_c.permute(0, 2, 1).contiguous()
        
        # 2. Quantize (Find the closest dictionary shapes)
        q_e, vq_loss_eeg, ids_e = self.eeg_vq(z_e)
        q_c, vq_loss_ecg, ids_c = self.ecg_vq(z_c)
        
        # Reshape back to (B, C, L) for Post-VQ Convolution
        q_e = q_e.permute(0, 2, 1).contiguous()
        q_c = q_c.permute(0, 2, 1).contiguous()
        
        q_e = self.eeg_post_vq(q_e)
        q_c = self.ecg_post_vq(q_c)
        
        # Reshape for Attention (B, L, C)
        q_e = q_e.permute(0, 2, 1).contiguous()
        q_c = q_c.permute(0, 2, 1).contiguous()
        
        # 3. Cross-Modal Attention (The secret weapon for sub-10% PRD)
        # EEG queries the ECG sequence (and vice-versa) to share multi-modal context
        attn_e, _ = self.cross_attn(query=q_e, key=q_c, value=q_c)
        attn_c, _ = self.cross_attn(query=q_c, key=q_e, value=q_e)
        
        # Add residual connection
        q_e = q_e + attn_e
        q_c = q_c + attn_c
        
        # Reshape back to (B, C, L)
        q_e = q_e.permute(0, 2, 1).contiguous()
        q_c = q_c.permute(0, 2, 1).contiguous()
        
        # 4. Decode
        rec_eeg = self.eeg_decoder(q_e)
        rec_ecg = self.ecg_decoder(q_c)
        
        return rec_eeg, rec_ecg, vq_loss_eeg, vq_loss_ecg
