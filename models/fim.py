import torch
import torch.nn as nn
import torch.nn.functional as F

class TextFIM(nn.Module):
    def __init__(self, d_model, snr_embed_dim=32, num_heads=8, dropout=0.1):
        super().__init__()
        # SNR embedding with better conditioning
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.LayerNorm(snr_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention for token relationships
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature importance prediction with sequence awareness
        self.importance_net = nn.Sequential(
            nn.Linear(d_model + snr_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, snr, target=None):
        # x shape: [batch_size, seq_len, d_model]
        # snr shape: [batch_size, 1] (normalized SNR values)
        batch_size, seq_len, d_model = x.shape
        
        # 1. Apply self-attention to capture token relationships
        attn_mask = None  # Can be set to handle padding if needed
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        
        # 2. Process SNR information
        snr_features = self.snr_embed(snr.unsqueeze(-1))  # [batch, snr_embed_dim]
        # Expand SNR features to match sequence length
        snr_features = snr_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, snr_embed_dim]
        
        # 3. Combine attention output with SNR features
        combined = torch.cat([attn_output, snr_features], dim=-1)  # [batch, seq_len, d_model+snr_embed_dim]
        
        # 4. Predict importance scores for each token
        importance_logits = self.importance_net(combined).squeeze(-1)  # [batch, seq_len]
        
        # 5. Compute FIM loss if in training mode
        fim_loss = None
        if self.training and target is not None:
            # For text data, we use token-level targets
            # Assuming target shape: [batch_size]
            # Create token-level targets where tokens matching the class are important
            token_targets = torch.zeros_like(importance_logits)
            
            # Set target tokens as important (this is a simplified approach)
            # In practice, you might have more sophisticated token-level importance labeling
            for i in range(batch_size):
                token_targets[i, target[i] % seq_len] = 1.0
                
            fim_loss = F.binary_cross_entropy_with_logits(
                importance_logits, 
                token_targets
            )
        
        # 6. Generate importance mask 
        importance_weights = torch.sigmoid(importance_logits).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # 7. Apply importance weights and refine features
        weighted_features = x * importance_weights
        refined_features = self.feature_refine(weighted_features)
        
        # 8. Add residual connection
        output = x + refined_features
        
        return output, fim_loss


class HiLoTextFIM(nn.Module):
    """
    High and Low Frequency decomposition-based Feature Importance Module for text.
    Inspired by Se-HiLo approach but adapted for text data.
    """
    def __init__(self, d_model, snr_embed_dim=32, num_heads=8, dropout=0.1, 
                 quantization_levels=8, noise_tolerance=0.5):
        super().__init__()
        self.d_model = d_model
        self.quantization_levels = quantization_levels
        self.noise_tolerance = noise_tolerance
        
        # SNR embedding
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.LayerNorm(snr_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # High-frequency extractor (captures fine-grained semantic details)
        self.high_freq_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Low-frequency extractor (captures coarse semantic structure)
        self.low_freq_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=1, padding=1),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # High-frequency importance network
        self.high_freq_importance = nn.Sequential(
            nn.Linear(d_model + snr_embed_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Low-frequency importance network
        self.low_freq_importance = nn.Sequential(
            nn.Linear(d_model + snr_embed_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Feature refinement with cross-modality fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def finite_scalar_quantization(self, x, levels):
        """
        Apply FSQ to improve noise resilience inspired by Se-HiLo.
        For text, we apply this to feature representations while preserving DeepSC dimensions.
        """
        # Keep original dimensions for proper residual connection
        orig_shape = x.shape
        
        # Reshape while preserving batch dimension
        x_reshaped = x.view(x.size(0), -1)
        
        # Keep dims for proper broadcasting
        x_min = x_reshaped.min(dim=-1, keepdim=True)[0]
        x_max = x_reshaped.max(dim=-1, keepdim=True)[0]
        
        # Scale to [0, levels-1] while preserving gradients
        x_scaled = (x_reshaped - x_min) / (x_max - x_min + 1e-6) * (levels - 1)
        
        # Quantize
        x_quantized = torch.round(x_scaled)
        
        # Rescale back to original range
        x_dequantized = x_quantized / (levels - 1) * (x_max - x_min + 1e-6) + x_min
        
        # Restore original shape
        return x_dequantized.view(orig_shape)
    
    def forward(self, x, snr, target=None):
        # x shape: [batch_size, seq_len, d_model]
        # snr shape: [batch_size, 1] (normalized SNR values)
        batch_size, seq_len, d_model = x.shape
        
        # 1. Process SNR information
        snr_value = snr.unsqueeze(-1)  # [batch, 1, 1]
        snr_features = self.snr_embed(snr_value)  # [batch, snr_embed_dim]
        
        # Adaptive quantization levels based on SNR
        # Higher SNR means we can use more levels (finer quantization)
        adaptive_levels = max(2, min(self.quantization_levels, 
                              int(self.quantization_levels * torch.sigmoid(snr_value / 10).item())))
        
        # 2. Extract high-frequency features (fine details)
        high_freq_out, _ = self.high_freq_attn(x, x, x)
        
        # 3. Extract low-frequency features (structural information)
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]
        low_freq_out = self.low_freq_conv(x_transposed)
        low_freq_out = low_freq_out.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # 4. Ensure low_freq_out has the same size as x
        if low_freq_out.size(1) != seq_len:
            low_freq_out = F.interpolate(
                low_freq_out.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # 5. Expand SNR features for both branches
        snr_features_expanded = snr_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 6. Compute importance scores for high and low frequency components
        high_freq_combined = torch.cat([high_freq_out, snr_features_expanded], dim=-1)
        low_freq_combined = torch.cat([low_freq_out, snr_features_expanded], dim=-1)
        
        high_freq_importance = self.high_freq_importance(high_freq_combined).sigmoid()
        low_freq_importance = self.low_freq_importance(low_freq_combined).sigmoid()
        
        # 7. Apply quantization to improve noise resilience
        # High frequency components get more quantization levels for detail preservation
        high_freq_quantized = self.finite_scalar_quantization(
            high_freq_out * high_freq_importance, 
            adaptive_levels
        )
        
        # Low frequency components need fewer levels but more robustness
        low_freq_quantized = self.finite_scalar_quantization(
            low_freq_out * low_freq_importance, 
            max(2, adaptive_levels // 2)
        )
        
        # 8. Combine high and low frequency components
        combined_features = torch.cat([high_freq_quantized, low_freq_quantized], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 9. Compute FIM loss if training
        fim_loss = None
        if self.training and target is not None:
            # Create token-level targets 
            token_targets = torch.zeros(batch_size, seq_len, device=x.device)
            
            # Set target tokens as important
            for i in range(batch_size):
                token_targets[i, target[i] % seq_len] = 1.0
            
            # Compute loss based on both high and low frequency importance
            high_freq_loss = F.binary_cross_entropy(
                high_freq_importance.squeeze(-1),
                token_targets
            )
            
            low_freq_loss = F.binary_cross_entropy(
                low_freq_importance.squeeze(-1),
                token_targets
            )
            
            fim_loss = high_freq_loss + low_freq_loss
        
        # 10. Add residual connection and return
        output = x + fused_features
        
        return output, fim_loss