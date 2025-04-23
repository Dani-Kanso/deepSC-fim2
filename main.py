# main.py
# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import SNR_to_noise, initNetParams, PowerNormalize, Channels
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from models.attack import TextFGSM  # Import from models instead of attacks

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-fim', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--gamma', default=0.3, type=float, help='FIM loss weight')
parser.add_argument('--if-attack-train', action='store_true')
parser.add_argument('--if-attack-test', action='store_true')
parser.add_argument('--train-snr', default=10, type=float)
parser.add_argument('--test-snr', default=10, type=float)
parser.add_argument('--print-freq', default=100, type=int)
parser.add_argument('--fim-type', default='standard', type=str, choices=['standard', 'hilo'],
                    help='Type of Feature Importance Module to use (standard or hilo)')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    print(f"Validating epoch {epoch}")
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, 
                               collate_fn=collate_data)
    net.eval()
    total_loss = 0
    total_samples = 0
    noise_std = SNR_to_noise(args.test_snr)
    
    with torch.no_grad():
        for sents, original_idx in tqdm(test_iterator, desc="Validation"):
            # Ensure tensors are LongTensor
            sents = sents.long().to(device)
            
            # Create source and target masks
            src_mask = (sents == pad_idx).unsqueeze(1).to(device)  # [B, 1, L]
            
            # Forward pass with FIM
            outputs, fim_loss = net(
                src=sents,
                trg_inp=sents,
                n_var=noise_std,
                src_mask=src_mask,
                trg_mask=src_mask,
                target=sents
            )
            
            # Compute task loss
            task_loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                sents.view(-1)
            )
            
            # Sum losses
            batch_loss = task_loss.item()
            if fim_loss is not None:
                # Make sure fim_loss is a scalar
                if isinstance(fim_loss, torch.Tensor) and fim_loss.numel() > 1:
                    fim_loss = fim_loss.mean()
                
                # Add to batch loss, checking if it's a tensor or float
                batch_loss += args.gamma * (fim_loss.item() if isinstance(fim_loss, torch.Tensor) else fim_loss)
                
            total_loss += batch_loss * sents.size(0)
            total_samples += sents.size(0)
    
    avg_loss = total_loss / total_samples
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss

def train(epoch, args, net, optimizer, mi_net=None, attack=None):
    print(f"Training epoch {epoch}")
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                collate_fn=collate_data)
    noise_std = SNR_to_noise(args.train_snr)
    
    net.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (sents, original_idx) in enumerate(tqdm(train_iterator, desc="Training")):
        # Ensure tensors are LongTensor
        sents = sents.long().to(device)
        optimizer.zero_grad()
        
        # Create source and target masks if needed
        src_mask = (sents == pad_idx).unsqueeze(1).to(device)  # [B, 1, L]
        
        # Apply adversarial attack if enabled
        if attack is not None:
            with torch.enable_grad():
                src_adv = attack.perturb(
                    sents, sents, noise_std, 
                    src_mask=src_mask, trg_mask=src_mask,
                    original_idx=original_idx  # Pass original indices to attack
                )
                # Ensure adversarial examples are also LongTensor
                src_adv = src_adv.long() if src_adv.dtype != torch.long else src_adv
        else:
            src_adv = sents
        
        # Forward pass with FIM
        outputs, fim_loss = net(
            src=src_adv,
            trg_inp=sents,
            n_var=noise_std,
            src_mask=src_mask,
            trg_mask=src_mask,
            target=sents
        )
        
        # Compute task loss
        task_loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            sents.view(-1)
        )
        
        # Total loss = task loss + gamma * FIM loss
        total_loss_value = task_loss
        if fim_loss is not None:
            # Make sure fim_loss is a scalar
            if isinstance(fim_loss, torch.Tensor) and fim_loss.numel() > 1:
                fim_loss = fim_loss.mean()
            total_loss_value = task_loss + args.gamma * fim_loss
        
        # Backward pass
        total_loss_value.backward()
        optimizer.step()
        
        # Log batch loss
        batch_loss = task_loss.item()
        if fim_loss is not None:
            # Make sure fim_loss is a scalar for logging
            if isinstance(fim_loss, torch.Tensor) and fim_loss.numel() > 1:
                fim_loss_scalar = fim_loss.mean()
            else:
                fim_loss_scalar = fim_loss
            batch_loss += args.gamma * (fim_loss_scalar.item() if isinstance(fim_loss_scalar, torch.Tensor) else fim_loss_scalar)
        
        total_loss += batch_loss * sents.size(0)
        total_samples += sents.size(0)
        
        # Print progress
        if (batch_idx + 1) % args.print_freq == 0:
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_iterator)}, Loss: {batch_loss:.4f}")
    
    avg_loss = total_loss / total_samples
    print(f"Training epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(10)
    
    print(f"Using FIM type: {args.fim_type}")
    print(f"Training with SNR: {args.train_snr}")
    print(f"Testing with SNR: {args.test_snr}")
    
    # Load vocabulary
    args.vocab_file = '/content/data/txt/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    
    # Set consistent max sequence length
    max_seq_len = 32  # Use fixed length to avoid dimension mismatches
    
    # Initialize models
    deepsc = DeepSC(
        num_layers=args.num_layers,
        src_vocab_size=num_vocab,
        trg_vocab_size=num_vocab,
        src_max_len=max_seq_len,  # Use consistent max length
        trg_max_len=max_seq_len,  # Use consistent max length
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        fim_type=args.fim_type
    ).to(device)
    
    print(f"Model initialized with {args.num_layers} layers, d_model={args.d_model}, heads={args.num_heads}")
    
    # Initialize optimizer and MI network
    mi_net = Mine().to(device)
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    
    # Initialize attack
    attack = None
    if args.if_attack_train:
        print("Initializing adversarial attack for training")
        attack = TextFGSM(deepsc, epsilon=0.1, alpha=0.01, return_embeddings=False)
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        print(f"Created checkpoint directory: {args.checkpoint_path}")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        train_loss = train(epoch, args, deepsc, optimizer, mi_net, attack)
        
        # Validate
        val_loss = validate(epoch, args, deepsc)
        
        # Save checkpoint
        checkpoint_path = f"{args.checkpoint_path}/epoch_{epoch+1}_loss_{val_loss:.4f}.pth"
        torch.save(deepsc.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"{args.checkpoint_path}/best_model.pth"
            torch.save(deepsc.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_path = f"{args.checkpoint_path}/final_checkpoint.pth"
    torch.save(deepsc.state_dict(), final_path)
    print(f"Training completed. Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")