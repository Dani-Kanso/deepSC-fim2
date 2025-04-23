import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.attack import TextFGSM
from models.metrics import FIMMetrics
from torch.utils.data import DataLoader
from utils import SNR_to_noise
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-fim', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--if-attack', action='store_true', help='Enable attack during evaluation')
parser.add_argument('--attack-epsilon', type=float, default=0.1, help='Attack strength')
parser.add_argument('--results-file', default='results.json', type=str)
parser.add_argument('--fim-type', default='standard', type=str, choices=['standard', 'hilo'],
                    help='Type of Feature Importance Module to use (standard or hilo)')
parser.add_argument('--max-seq-len', default=32, type=int, help='Maximum sequence length to match training')

def evaluate(args, model, metrics, dataloader, snr_range):
    """Evaluate model across different SNR values"""
    results = {snr: {'clean': {}, 'attacked': {}} for snr in snr_range}
    model.eval()
    
    # Initialize attack if needed
    attack = None
    if args.if_attack:
        attack = TextFGSM(model, epsilon=args.attack_epsilon, return_embeddings=False)
    
    with torch.no_grad():
        for snr in snr_range:
            print(f"\nEvaluating SNR: {snr} dB")
            noise_var = SNR_to_noise(snr)
            
            # Track metrics for this SNR
            batch_metrics = []
            
            for sents, original_idx in tqdm(dataloader):
                # Move batch to device and handle masks
                sents = sents.to(device)
                src_mask = (sents == token_to_idx["<PAD>"]).unsqueeze(-2)
                
                if attack is not None:
                    # For evaluation under attack, create simple perturbations without requiring gradients
                    batch_result = metrics.evaluate_batch_with_attack(
                        model=model, 
                        clean_input=sents, 
                        noise_var=noise_var,
                        attack_epsilon=args.attack_epsilon
                    )
                else:
                    batch_result = metrics.evaluate_batch(
                        model=model, 
                        clean_input=sents, 
                        attacked_input=None, 
                        noise_var=noise_var
                    )
                
                batch_metrics.append(batch_result)
            
            # Average metrics across batches
            snr_metrics = {}
            for key in batch_metrics[0].keys():
                values = [m[key] for m in batch_metrics]
                snr_metrics[key] = float(np.mean(values))
            
            # Organize results
            if args.if_attack:
                results[snr]['clean'] = {
                    'bleu1': snr_metrics['bleu1'],
                    'bleu4': snr_metrics['bleu4']
                }
                results[snr]['attacked'] = {
                    'bleu1': snr_metrics['attacked_bleu1'],
                    'bleu4': snr_metrics['attacked_bleu4'],
                    'asr': snr_metrics['asr'],
                    'fis': snr_metrics['fis'],
                    'tdr': snr_metrics['tdr']
                }
            else:
                results[snr]['clean'] = {
                    'bleu1': snr_metrics['bleu1'],
                    'bleu4': snr_metrics['bleu4']
                }
            
            # Print current SNR results
            print(f"\nSNR {snr} dB Results:")
            print(f"Clean BLEU-1: {results[snr]['clean']['bleu1']:.4f}")
            print(f"Clean BLEU-4: {results[snr]['clean']['bleu4']:.4f}")
            if args.if_attack:
                print(f"Attacked BLEU-1: {results[snr]['attacked']['bleu1']:.4f}")
                print(f"Attack Success Rate: {results[snr]['attacked']['asr']:.4f}")
                print(f"Feature Importance Stability: {results[snr]['attacked']['fis']:.4f}")
    
    return results

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix the vocabulary file path to match the training path structure
    args.vocab_file = '/content/data/txt/' + args.vocab_file
    
    # Load vocabulary
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    
    # Use fixed max_seq_len to match the training model
    max_seq_len = args.max_seq_len
    
    # Initialize model with same parameters as training
    model = DeepSC(
        num_layers=args.num_layers,
        src_vocab_size=num_vocab,
        trg_vocab_size=num_vocab,
        src_max_len=max_seq_len,
        trg_max_len=max_seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=0.1,
        fim_type=args.fim_type
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)
    print('Model loaded successfully!')
    
    # Initialize metrics
    metrics = FIMMetrics(token_to_idx, device)
    
    # Prepare data
    test_dataset = EurDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           num_workers=0, pin_memory=True,
                           collate_fn=collate_data)
    
    # Define SNR range
    snr_range = [0, 3, 6, 9, 12, 15, 18]
    
    # Run evaluation
    results = evaluate(args, model, metrics, test_loader, snr_range)
    
    # Save results
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.results_file}")