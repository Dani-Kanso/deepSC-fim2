import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from w3lib.html import remove_tags

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
    
    def evaluate_batch(self, model, batch, attack=None, noise_var=0.1):
        """Evaluate a batch with all metrics"""
        # Handle batch as a single tensor since collate_data returns a single tensor
        sents = batch.to(self.device)
        
        # Create masks
        src_mask = (sents == self.vocab['<PAD>']).unsqueeze(-2)
        trg_inp = sents[:, :-1]
        trg_real = sents[:, 1:]
        
        # Get clean predictions
        clean_output = model(sents, trg_inp, src_mask, None, noise_var, target=trg_real)
        clean_preds = clean_output['out_x']
        clean_scores = clean_output['importance_scores']
        
        metrics = {
            'bleu1': self.calculate_bleu(clean_preds, trg_real),
            'bleu4': self.calculate_bleu(clean_preds, trg_real, weights=(0.25, 0.25, 0.25, 0.25))
        }
        
        if attack is not None:
            # Get predictions under attack
            attacked_output = model(sents, trg_inp, src_mask, None, noise_var, target=trg_real)
            attacked_preds = attacked_output['out_x']
            attacked_scores = attacked_output['importance_scores']
            
            metrics.update({
                'asr': self.calculate_asr(clean_preds, attacked_preds, trg_real, self.vocab['<PAD>']),
                'fis': self.calculate_fis(clean_scores, attacked_scores, src_mask),
                'attacked_bleu1': self.calculate_bleu(attacked_preds, trg_real),
                'attacked_bleu4': self.calculate_bleu(attacked_preds, trg_real, weights=(0.25, 0.25, 0.25, 0.25))
            })
            metrics['tdr'] = self.calculate_tdr(metrics['bleu1'], metrics['attacked_bleu1'])
        
        return metrics 