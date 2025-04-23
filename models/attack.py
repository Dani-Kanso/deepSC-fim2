import torch
import torch.nn.functional as F

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        x = torch.clamp(x, original_x - epsilon, original_x + epsilon)
    elif _type == 'l2':
        delta = x - original_x
        delta = delta * torch.min(torch.tensor(1.0), epsilon / torch.norm(delta, p=2))
        x = original_x + delta
    return x

class TextFGSM:
    def __init__(self, model, epsilon=0.1, alpha=0.01, min_val=None, max_val=None, max_iters=1):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iters = max_iters
        self.min_val = min_val
        self.max_val = max_val

    def perturb(self, original_embeddings, target, n_var, src_mask=None, trg_mask=None, 
                reduction4loss='mean', random_start=False, beta=1.0, _type='linf'):
        """
        Generate adversarial examples using FGSM
        
        Args:
            original_embeddings: Input embeddings to perturb
            target: Target labels
            n_var: Noise variance
            src_mask: Source padding mask (optional)
            trg_mask: Target padding mask (optional)
            reduction4loss: Loss reduction method
            random_start: Whether to use random starting point
            beta: Weight for FIM loss
            _type: Type of perturbation constraint ('linf' or 'l2')
            
        Returns:
            Perturbed embeddings
        """
        x = original_embeddings.clone()
        if random_start:
            x += torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            if self.min_val is not None and self.max_val is not None:
                x.clamp_(self.min_val, self.max_val)

        for _ in range(self.max_iters):
            x = x.detach().requires_grad_()
            
            # Forward through full model
            outputs, fim_loss = self.model(
                src=x, 
                trg_inp=target, 
                n_var=n_var, 
                src_mask=src_mask,
                trg_mask=trg_mask,
                target=target
            )
            
            # Compute task loss + FIM loss
            task_loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                target.view(-1), 
                reduction=reduction4loss
            )
            
            # Handle case where fim_loss might be None or a tensor
            if fim_loss is not None:
                total_loss = task_loss + beta * fim_loss
            else:
                total_loss = task_loss
            
            # Compute gradient
            grad = torch.autograd.grad(total_loss, x)[0]
            
            # Update perturbation
            with torch.no_grad():
                x = x + self.alpha * torch.sign(grad)
                x = project(x, original_embeddings, self.epsilon, _type)
                if self.min_val is not None and self.max_val is not None:
                    x.clamp_(self.min_val, self.max_val)
        
        return x