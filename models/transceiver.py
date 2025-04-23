# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
这是一个Transformer的网络结构
"""
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from .fim import TextFIM, HiLoTextFIM
from utils import PowerNormalize, Channels


class PositionalEncoding(nn.Module):
    "Implement the PE function with dynamic sequence length handling."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-compute position embeddings for the maximum length
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # If sequence is longer than buffer, create new positional encodings
            pe = torch.zeros(1, seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * 
                               -(math.log(10000.0) / self.d_model))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            # Use pre-computed encodings
            pe = self.pe[:, :seq_len]
        
        x = x + pe
        return self.dropout(x)
  
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

#class LayerNorm(nn.Module):
#    "Construct a layernorm module (See citation for details)."
#    # features = d_model
#    def __init__(self, features, eps=1e-6):
#        super(LayerNorm, self).__init__()
#        self.a_2 = nn.Parameter(torch.ones(features))
#        self.b_2 = nn.Parameter(torch.zeros(features))
#        self.eps = eps
#
#    def forward(self, x):
#        mean = x.mean(-1, keepdim=True)
#        std = x.std(-1, keepdim=True)
#        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x
        


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output
        

class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, 
                 src_max_len, trg_max_len, d_model, num_heads, dff, dropout=0.1, fim_type='standard'):
        super().__init__()
        self.d_model = d_model
        self.fim_type = fim_type
        
        # Encoder components
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.encoder_pe = PositionalEncoding(d_model, dropout, src_max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])
        
        # Initialize FIM encoders based on fim_type
        self.fim_encoders = nn.ModuleList()
        for _ in range(num_layers):
            if fim_type == 'hilo':
                self.fim_encoders.append(HiLoTextFIM(d_model, num_heads=num_heads, dropout=dropout))
            else:  # default to standard
                self.fim_encoders.append(TextFIM(d_model, num_heads=num_heads, dropout=dropout))
        
        # Decoder components
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.decoder_pe = PositionalEncoding(d_model, dropout, trg_max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])
        
        # Initialize FIM decoders based on fim_type
        self.fim_decoders = nn.ModuleList()
        for _ in range(num_layers):
            if fim_type == 'hilo':
                self.fim_decoders.append(HiLoTextFIM(d_model, num_heads=num_heads, dropout=dropout))
            else:  # default to standard
                self.fim_decoders.append(TextFIM(d_model, num_heads=num_heads, dropout=dropout))
        
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))

        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.dense = nn.Linear(d_model, trg_vocab_size)
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def calculate_snr(self, n_var):
        """Calculate SNR from noise variance"""
        if isinstance(n_var, float) or isinstance(n_var, int):
            # Convert noise variance to dB SNR
            # SNR = 10 * log10(1/n_var) for normalized signal power of 1
            return 10 * math.log10(1/n_var)
        else:
            # Handle tensor case
            return 10 * torch.log10(1/n_var)

    def forward(self, src, trg_inp, n_var, src_mask=None, trg_mask=None, target=None):
        fim_loss_total = 0.0
        snr = self.calculate_snr(n_var)
        
        # Encoder with per-layer FIM
        # Check if src is already an embedding tensor (output from TextFGSM.perturb)
        if src.dim() == 3 and src.size(2) == self.d_model:
            # src is already an embedding tensor
            enc_output = src
        else:
            # src is a token index tensor, needs embedding
            src = src.long() if src.dtype != torch.long else src
            enc_output = self.encoder_embedding(src) * math.sqrt(self.d_model)
            enc_output = self.encoder_pe(enc_output)
        
        for enc_layer, fim in zip(self.encoder_layers, self.fim_encoders):
            enc_output = enc_layer(enc_output, src_mask)
            enc_output, fim_loss = fim(enc_output, snr, mask=src_mask, target=target)
            fim_loss_total += fim_loss if fim_loss is not None else 0.0
        
        # Channel processing (keep your existing implementation)
        channel_enc_output = self.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)
        Rx_sig = Channels().AWGN(Tx_sig, n_var)
        channel_dec_output = self.channel_decoder(Rx_sig)
        
        # Decoder with per-layer FIM
        dec_output = self.decoder_embedding(trg_inp) * math.sqrt(self.d_model)
        dec_output = self.decoder_pe(dec_output)
        
        for dec_layer, fim in zip(self.decoder_layers, self.fim_decoders):
            dec_output = dec_layer(
                dec_output, channel_dec_output,
                trg_mask, src_mask
            )
            dec_output, fim_loss = fim(dec_output, snr, mask=trg_mask, target=target)
            fim_loss_total += fim_loss if fim_loss is not None else 0.0
        
        # Final output
        output = self.fc_out(dec_output)
        
        # Ensure fim_loss_total is a scalar
        if isinstance(fim_loss_total, torch.Tensor) and fim_loss_total.numel() > 1:
            fim_loss_total = fim_loss_total.mean()
        
        return output, fim_loss_total

    def forward_from_embeddings(self, embeddings, trg_inp, n_var, src_mask=None, trg_mask=None, target=None):
        """Forward pass starting from embeddings instead of raw tokens"""
        fim_loss_total = 0.0
        snr = self.calculate_snr(n_var)
        
        # Start from provided embeddings
        enc_output = embeddings
        
        # Apply FIM layers
        for fim in self.fim_encoders:
            enc_output, fim_loss = fim(
                enc_output,
                snr,
                mask=src_mask,  # Pass src_mask to FIM
                target=target
            )
            fim_loss_total += fim_loss if fim_loss is not None else 0.0
        
        # Channel encoding and transmission
        channel_enc_output = self.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)
        
        # Channel transmission
        Rx_sig = Channels().AWGN(Tx_sig, n_var)
        
        # Channel decoding
        channel_dec_output = self.channel_decoder(Rx_sig)
        
        # Decoder processing with proper masking
        # Ensure trg_inp is LongTensor
        trg_inp = trg_inp.long() if trg_inp.dtype != torch.long else trg_inp
        dec_output = self.decoder_embedding(trg_inp) * math.sqrt(self.d_model)
        dec_output = self.decoder_pe(dec_output)
        
        for dec_layer, fim in zip(self.decoder_layers, self.fim_decoders):
            dec_output = dec_layer(
                dec_output, channel_dec_output,
                trg_mask, src_mask
            )
            dec_output, fim_loss = fim(
                dec_output,
                snr,
                mask=trg_mask,
                target=target
            )
            fim_loss_total += fim_loss if fim_loss is not None else 0.0
        
        # Final output
        output = self.fc_out(dec_output)
        
        # Ensure fim_loss_total is a scalar
        if isinstance(fim_loss_total, torch.Tensor) and fim_loss_total.numel() > 1:
            fim_loss_total = fim_loss_total.mean()
        
        return output, fim_loss_total




















