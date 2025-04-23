import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextFIM(nn.Module):
    def __init__(self, d_model, num_heads, dff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention for feature importance
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed forward network
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feature importance classifier
        self.fc = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, 1)  # Binary importance score per token
        self.relu = nn.ReLU()

    def forward(self, x, snr=None, mask=None, target=None):
        # Apply attention and FFN
        attn_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        x = self.relu(x)
        
        # Get feature importance scores
        features = self.fc(x)
        importance_scores = self.classifier(features).squeeze(-1)  # [batch_size, seq_len]
        
        # FIM loss - this will be scalar
        fim_loss = None
        if self.training and target is not None:
            # During training, mask features based on target class
            fim_loss = torch.mean(torch.abs(importance_scores))
            
            # Apply feature masking
            importance_mask = torch.sigmoid(importance_scores)
            x = x * importance_mask.unsqueeze(-1)
        else:
            # During inference, mask features based on predicted importance
            importance_mask = torch.sigmoid(importance_scores)
            x = x * importance_mask.unsqueeze(-1)
            
        return x, fim_loss

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        
        # Only unsqueeze mask if it's not None
        if mask is not None:
            # Check if mask is a tensor before attempting to unsqueeze
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(1)
        
        # Linear projections
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None and isinstance(mask, torch.Tensor):
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        x = self.dense(x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, dff)
        self.w2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class HiLoTextFIM(nn.Module):
    def __init__(self, d_model, num_heads=8, dff=512, dropout=0.1, 
                 quantization_levels=8, snr_embed_dim=32):
        super().__init__()
        self.d_model = d_model
        self.quantization_levels = quantization_levels
        
        # Input normalization
        self.norm_input = nn.LayerNorm(d_model, eps=1e-6)
        
        # SNR embedding with normalization
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.LayerNorm(snr_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.snr_norm = nn.LayerNorm(snr_embed_dim, eps=1e-6)
        
        # High-frequency path
        self.high_freq_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.high_freq_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Low-frequency path
        self.low_freq_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.low_freq_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Feature importance networks
        combined_dim = d_model + snr_embed_dim
        self.high_freq_importance = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.low_freq_importance = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output normalization
        self.norm_output = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, snr, mask=None, target=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            snr: SNR values [batch_size, 1]
            mask: Optional attention mask
            target: Optional target tensor for training
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Input normalization
        x = self.norm_input(x)
        
        # 2. Process SNR information
        snr_value = snr.to(dtype=torch.float32) if isinstance(snr, torch.Tensor) else torch.tensor(snr, dtype=torch.float32, device=x.device)
        if snr_value.dim() == 0:
            snr_value = snr_value.unsqueeze(0)
        if snr_value.dim() == 1:
            snr_value = snr_value.unsqueeze(-1)  # [batch_size, 1]
            
        # Get SNR embedding and normalize
        snr_features = self.snr_embed(snr_value)  # [batch_size, snr_embed_dim]
        snr_features = self.snr_norm(snr_features)
        
        # Expand SNR features to match sequence length
        snr_features = (snr_features
            .unsqueeze(1)  # [batch_size, 1, snr_embed_dim]
            .expand(-1, seq_len, -1)  # [batch_size, seq_len, snr_embed_dim]
        )
        
        # 3. High-frequency path (with mask if provided)
        if mask is not None and isinstance(mask, torch.Tensor):
            # Convert mask to attention mask format expected by PyTorch attention
            attn_mask = (1.0 - mask.float()) * -10000.0 if mask.dim() == 3 else None
            high_freq_out, _ = self.high_freq_attn(x, x, x, attn_mask=attn_mask if attn_mask is not None else None)
        else:
            high_freq_out, _ = self.high_freq_attn(x, x, x)
        high_freq_out = self.high_freq_norm(high_freq_out)
        
        # 4. Low-frequency path
        low_freq_out = self.low_freq_conv(x.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len, d_model]
        low_freq_out = self.low_freq_norm(low_freq_out)
        
        # 5. Compute importance scores
        high_freq_combined = torch.cat([high_freq_out, snr_features], dim=-1)
        low_freq_combined = torch.cat([low_freq_out, snr_features], dim=-1)
        
        high_importance = self.high_freq_importance(high_freq_combined)  # [batch_size, seq_len, 1]
        low_importance = self.low_freq_importance(low_freq_combined)  # [batch_size, seq_len, 1]
        
        # 6. Apply importance weighting
        high_freq_weighted = high_freq_out * high_importance
        low_freq_weighted = low_freq_out * low_importance
        
        # 7. Adaptive quantization based on SNR
        quantization_level = max(2, int(self.quantization_levels * torch.sigmoid(snr_value.mean()).item()))
        high_freq_weighted = self.finite_scalar_quantization(high_freq_weighted.contiguous(), quantization_level)
        low_freq_weighted = self.finite_scalar_quantization(low_freq_weighted.contiguous(), max(2, quantization_level // 2))
        
        # 8. Feature fusion
        combined_features = torch.cat([high_freq_weighted, low_freq_weighted], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 9. Compute FIM loss if training
        fim_loss = None
        if self.training and target is not None:
            token_targets = torch.zeros(batch_size, seq_len, device=x.device)
            for i in range(batch_size):
                if target[i] is not None:
                    idx = min(int(target[i].item() if torch.is_tensor(target[i]) else target[i]), seq_len - 1)
                    token_targets[i, idx] = 1.0
                    
            high_freq_loss = F.binary_cross_entropy(
                high_importance.squeeze(-1),
                token_targets
            )
            
            low_freq_loss = F.binary_cross_entropy(
                low_importance.squeeze(-1),
                token_targets
            )
            
            fim_loss = high_freq_loss + low_freq_loss
        
        # 10. Final normalization and residual connection
        output = self.norm_output(x + fused_features)
        
        return output, fim_loss
        
    def finite_scalar_quantization(self, x, levels):
        """Apply FSQ with proper tensor reshaping"""
        # Store original shape
        orig_shape = x.shape
        
        # Ensure tensor is contiguous and reshape
        x = x.contiguous()
        x_reshaped = x.reshape(x.size(0), -1)
        
        # Get min/max per batch
        x_min = x_reshaped.min(dim=1, keepdim=True)[0]
        x_max = x_reshaped.max(dim=1, keepdim=True)[0]
        
        # Scale to [0, levels-1]
        x_range = x_max - x_min
        x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)  # Prevent division by zero
        x_scaled = (x_reshaped - x_min) / x_range * (levels - 1)
        
        # Quantize
        x_quantized = torch.round(x_scaled)
        
        # Scale back
        x_dequantized = x_quantized / (levels - 1) * x_range + x_min
        
        # Restore original shape
        return x_dequantized.reshape(orig_shape)