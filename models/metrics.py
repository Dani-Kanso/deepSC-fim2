import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from w3lib.html import remove_tags
import math

class FIMMetrics:
    def __init__(self, vocab, device='cuda'):
        self.vocab = vocab
        self.device = device
        self.special_tokens = {'<PAD>', '<START>', '<END>'}
        self.special_ids = {vocab[token] for token in self.special_tokens if token in vocab}
        
    def idx2words(self, idx_list):
        """Convert token indices to words, removing special tokens"""
        words = []
        idx_to_token = {v: k for k, v in self.vocab.items()}
        for idx in idx_list:
            if idx in self.special_ids:
                continue
            words.append(idx_to_token.get(idx, '<UNK>'))
        return words
    
    def calculate_bleu(self, predictions, targets, weights=(1.0, 0, 0, 0)):
        """Calculate BLEU score for batched predictions"""
        scores = []
        pred_tokens = predictions.argmax(dim=-1).cpu().numpy()
        target_tokens = targets.cpu().numpy()
        
        for pred, target in zip(pred_tokens, target_tokens):
            pred_words = self.idx2words(pred)
            target_words = self.idx2words(target)
            if pred_words and target_words:  # Only calculate if both sequences have valid words
                score = sentence_bleu([target_words], pred_words, weights=weights)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def calculate_asr(self, clean_preds, attacked_preds, targets, pad_idx):
        """Calculate Attack Success Rate"""
        # Create mask for padding
        mask = (targets != pad_idx)
        
        # Get predictions
        clean_pred = clean_preds.argmax(dim=-1)
        attacked_pred = attacked_preds.argmax(dim=-1)
        
        # Calculate success rate only on non-padded tokens
        clean_correct = ((clean_pred == targets) * mask).float()
        attacked_wrong = ((attacked_pred != targets) * mask).float()
        
        asr = (clean_correct * attacked_wrong).sum() / mask.sum()
        return asr.item()
    
    def calculate_fis(self, clean_scores, attacked_scores, src_mask=None):
        """Calculate Feature Importance Stability using cosine similarity"""
        if src_mask is not None:
            # Only consider non-masked positions
            clean_scores = clean_scores.masked_fill(src_mask, 0)
            attacked_scores = attacked_scores.masked_fill(src_mask, 0)
        
        # Normalize scores
        clean_norm = torch.sigmoid(clean_scores)
        attacked_norm = torch.sigmoid(attacked_scores)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(clean_norm, attacked_norm, dim=-1)
        return similarity.mean().item()
    
    def calculate_tdr(self, clean_bleu, attacked_bleu):
        """Calculate Translation Degradation Rate"""
        if clean_bleu == 0:
            return 1.0  # Maximum degradation if clean translation was already bad
        return (clean_bleu - attacked_bleu) / clean_bleu
    
    def evaluate_batch(self, model, clean_input, attacked_input=None, noise_var=0.1):
        """Evaluate a batch with all metrics"""
        # Ensure inputs are on the correct device
        sents = clean_input.to(self.device)
        
        # Create masks
        src_mask = (sents == self.vocab['<PAD>']).unsqueeze(-2)
        
        # Forward pass with clean inputs
        with torch.no_grad():
            clean_outputs, clean_fim_loss = model(
                src=sents,
                trg_inp=sents,
                n_var=noise_var,
                src_mask=src_mask,
                trg_mask=src_mask,
                target=sents
            )
        
        # Calculate BLEU scores for clean predictions
        metrics = {
            'bleu1': self.calculate_bleu(clean_outputs, sents, weights=(1.0, 0, 0, 0)),
            'bleu4': self.calculate_bleu(clean_outputs, sents, weights=(0.25, 0.25, 0.25, 0.25))
        }
        
        # If attacked input is provided, calculate adversarial metrics
        if attacked_input is not None:
            attacked_sents = attacked_input.to(self.device)
            
            # Forward pass with attacked inputs
            with torch.no_grad():
                attacked_outputs, _ = model(
                    src=attacked_sents,
                    trg_inp=sents,  # Still use clean inputs for target
                    n_var=noise_var,
                    src_mask=src_mask,
                    trg_mask=src_mask,
                    target=sents
                )
            
            # Get importance scores from the model's FIM layers (using a simple approximation)
            # Note: This is a simplified approach as we don't have direct access to importance scores
            
            # Calculate attack success metrics
            metrics.update({
                'attacked_bleu1': self.calculate_bleu(attacked_outputs, sents, weights=(1.0, 0, 0, 0)),
                'attacked_bleu4': self.calculate_bleu(attacked_outputs, sents, weights=(0.25, 0.25, 0.25, 0.25)),
                'asr': self.calculate_asr_simple(clean_outputs, attacked_outputs, sents),
                'fis': 0.5,  # Placeholder since we don't have importance scores directly
                'tdr': self.calculate_tdr_simple(
                    self.calculate_bleu(clean_outputs, sents),
                    self.calculate_bleu(attacked_outputs, sents)
                )
            })
        
        return metrics

    def evaluate_batch_with_attack(self, model, clean_input, noise_var=0.1, attack_epsilon=0.1):
        """Evaluate a batch with a simple adversarial attack that doesn't require gradients"""
        # Ensure inputs are on the correct device
        sents = clean_input.to(self.device)
        
        # Create masks
        src_mask = (sents == self.vocab['<PAD>']).unsqueeze(-2)
        
        # Forward pass with clean inputs
        with torch.no_grad():
            clean_outputs, clean_fim_loss = model(
                src=sents,
                trg_inp=sents,
                n_var=noise_var,
                src_mask=src_mask,
                trg_mask=src_mask,
                target=sents
            )
            
            # For adversarial evaluation, get embeddings first
            embeddings = model.encoder_embedding(sents) * math.sqrt(model.d_model)
            embeddings = model.encoder_pe(embeddings)
            
            # Create a simple perturbation directly in embedding space
            noise = torch.randn_like(embeddings) * attack_epsilon
            perturbed_embeddings = embeddings + noise
            
            # Forward pass with perturbed embeddings
            attacked_outputs, _ = model.forward_from_embeddings(
                embeddings=perturbed_embeddings,
                trg_inp=sents,
                n_var=noise_var,
                src_mask=src_mask,
                trg_mask=src_mask,
                target=sents
            )
        
        # Calculate metrics
        metrics = {
            'bleu1': self.calculate_bleu(clean_outputs, sents, weights=(1.0, 0, 0, 0)),
            'bleu4': self.calculate_bleu(clean_outputs, sents, weights=(0.25, 0.25, 0.25, 0.25)),
            'attacked_bleu1': self.calculate_bleu(attacked_outputs, sents, weights=(1.0, 0, 0, 0)),
            'attacked_bleu4': self.calculate_bleu(attacked_outputs, sents, weights=(0.25, 0.25, 0.25, 0.25)),
            'asr': self.calculate_asr_simple(clean_outputs, attacked_outputs, sents),
            'fis': 0.5,  # Placeholder for feature importance stability
            'tdr': self.calculate_tdr_simple(
                self.calculate_bleu(clean_outputs, sents),
                self.calculate_bleu(attacked_outputs, sents)
            )
        }
        
        return metrics
        
    def calculate_asr_simple(self, clean_outputs, attacked_outputs, targets):
        """Simple Attack Success Rate calculation for evaluation"""
        # Get predictions
        clean_pred = clean_outputs.argmax(dim=-1)
        attacked_pred = attacked_outputs.argmax(dim=-1)
        
        # Calculate how many predictions changed after attack
        changed = (clean_pred != attacked_pred).float().mean()
        return changed.item()
        
    def calculate_tdr_simple(self, clean_bleu, attacked_bleu):
        """Calculate Translation Degradation Rate"""
        if clean_bleu == 0:
            return 1.0  # Maximum degradation if clean translation was already bad
        return (clean_bleu - attacked_bleu) / clean_bleu 