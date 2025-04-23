import torch
import torch.nn.functional as F
import math

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        x = torch.clamp(x, original_x - epsilon, original_x + epsilon)
    elif _type == 'l2':
        delta = x - original_x
        delta = delta * torch.min(torch.tensor(1.0), epsilon / torch.norm(delta, p=2))
        x = original_x + delta
    return x

class TextFGSM:
    def __init__(self, model, epsilon=0.1, alpha=0.01, min_val=None, max_val=None, max_iters=1, return_embeddings=False):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iters = max_iters
        self.min_val = min_val
        self.max_val = max_val
        self.return_embeddings = return_embeddings
    
    def get_embeddings(self, tokens):
        """Helper method to get embeddings while ensuring consistent dimensions"""
        batch_size, seq_len = tokens.shape
        
        # Get initial embeddings
        with torch.no_grad():
            # Get embeddings and scale
            embeddings = self.model.encoder_embedding(tokens) * (self.model.d_model ** 0.5)
            
            # Get positional encoding and ensure it matches sequence length
            pos_encoding = self.model.encoder_pe.pe
            if seq_len > pos_encoding.size(1):
                # If sequence is longer than pe buffer, recreate pe for this length
                device = embeddings.device
                pe = torch.zeros(1, seq_len, self.model.d_model, device=device)
                position = torch.arange(0, seq_len, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.model.d_model, 2, device=device) * 
                                   -(math.log(10000.0) / self.model.d_model))
                pe[0, :, 0::2] = torch.sin(position * div_term)
                pe[0, :, 1::2] = torch.cos(position * div_term)
                pos_encoding = pe
            
            # Add positional encoding
            embeddings = embeddings + pos_encoding[:, :seq_len].to(embeddings.device)
            embeddings = self.model.encoder_pe.dropout(embeddings)
            
        return embeddings

    def perturb(self, tokens, target, n_var, src_mask=None, trg_mask=None, 
                reduction4loss='mean', random_start=False, beta=1.0, _type='linf', original_idx=None):
        """Generate adversarial examples by perturbing embeddings"""
        # Check if we're in evaluation mode (no_grad)
        is_eval_mode = not torch.is_grad_enabled()
        
        # If in evaluation mode and not returning embeddings, just return tokens
        if is_eval_mode and not self.return_embeddings:
            return tokens
            
        # Get initial embeddings
        embeddings = self.get_embeddings(tokens)
        
        # If in evaluation mode and returning embeddings, apply a simple perturbation without gradients
        if is_eval_mode and self.return_embeddings:
            # Create a simple non-gradient based perturbation for evaluation
            noise = torch.randn_like(embeddings) * self.epsilon
            perturbed = embeddings + noise
            return perturbed
        
        # Regular training mode with gradients below
        # Initialize perturbation
        x = embeddings.clone()
        if random_start:
            x += torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            if self.min_val is not None and self.max_val is not None:
                x.clamp_(self.min_val, self.max_val)

        # Create inverse mapping for original indices if provided
        if original_idx is not None:
            inv_idx = torch.empty(len(original_idx), dtype=torch.long, device=tokens.device)
            inv_idx[torch.tensor(original_idx, device=tokens.device)] = torch.arange(len(original_idx), device=tokens.device)
        
        for _ in range(self.max_iters):
            x = x.detach().requires_grad_()
            
            # Forward pass
            outputs, fim_loss = self.model.forward_from_embeddings(
                embeddings=x,
                trg_inp=target,
                n_var=n_var,
                src_mask=src_mask,
                trg_mask=trg_mask,
                target=target
            )
            
            # Loss computation - ensure reduction to scalar
            task_loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                target.reshape(-1),
                reduction='mean'  # Always use mean reduction for gradients
            )
            
            # Sum losses ensuring scalar output
            total_loss = task_loss
            if fim_loss is not None:
                # If fim_loss is a tensor, ensure it's reduced to scalar
                if isinstance(fim_loss, torch.Tensor):
                    fim_loss = fim_loss.mean()
                total_loss = task_loss + beta * fim_loss
            
            # Gradient computation
            grad = torch.autograd.grad(total_loss, x)[0]
            
            with torch.no_grad():
                # Calculate update with proper sequence ordering
                update = self.alpha * torch.sign(grad)
                if original_idx is not None:
                    # Reorder update according to original sequence ordering
                    update = update[inv_idx]
                
                x = x + update
                x = project(x, embeddings, self.epsilon, _type)
                if self.min_val is not None and self.max_val is not None:
                    x.clamp_(self.min_val, self.max_val)
        
        # Return perturbed embeddings or original tokens based on return_embeddings flag
        if self.return_embeddings:
            return x
        else:
            # If we're not returning embeddings, return the original tokens
            # The model will handle the conversion in forward method
            return tokens