import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from function import normal,normal_style
import numpy as np
import os
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=5,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayerWithSAAM(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.linear_proj = nn.Linear(256, self.d_model)
        self.linear_proj2 = nn.Linear(self.d_model, 256)        
        
        self.new_ps = nn.Conv2d(512 , 512 , (1,1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, resnet_output): # [256, 32, 32]，[256, 32, 32]
        '''
        style = style.permute(1, 2, 0)  # [1, 512, 512, 3]
        style = style.reshape(1, -1, 3)  # [1, 512*512, 3]
        content = content.permute(1, 2, 0)  # [1, 512, 512, 3]
        content = content.reshape(1, -1, 3)  # [1, 512*512, 3]

        style = self.linear_proj(style)  # [1, 512*512, d_model]
        content = self.linear_proj(content)  # [1, 512*512, d_model]
        '''
        if resnet_output.shape[0] == 2:
            style = resnet_output[1, :]
            content = resnet_output[0, :]

            content = content.permute(1, 2, 0)# [32, 32, 256]
            content = content.reshape(-1, 256)  # [ 32*32, 256]

            content = self.linear_proj(content)  # [1024, 512]

            content = content.unsqueeze(0)


            #[1, 1024, 512]-->[256, 32, 32]
            #if 1, style img feed in the style transfer net
            Semantic_aware_loss = 1 # 1 is to calculate the semantic perception loss
            if Semantic_aware_loss == 1:
                style = style.permute(1, 2, 0)# [32, 32, 256]
                style = style.reshape(-1, 256)  # [ 32*32, 256]
                style = self.linear_proj(style)  # [1024, 512]
                style = style.unsqueeze(0)
                style = self.encoder_s(style)           # [1, 1024, 512]
                style = self.linear_proj2(style) #[1, 1024, 256]
                style = style.squeeze(0)#[1024, 256]
                r = int(np.sqrt(style.shape[0]))
                style = style.reshape(r,r,256)# [32, 32, 256]
                style = style.permute(2, 0, 1)# [256, 32, 32]

            content = self.encoder_c(content)       # [1, 1024, 512]
            content = self.linear_proj2(content) #[1, 1024, 256]
            content = content.squeeze(0)#[1024, 256]
            r = int(np.sqrt(content.shape[0]))
            content = content.reshape(r,r,256)# [32, 32, 256]
            content = content.permute(2, 0, 1)# [256, 32, 32]
            out = torch.stack([content, style], dim=0)

        elif resnet_output.shape[0] == 1:
            img = resnet_output.squeeze(0) #img[256, 2, 2]
            img = img.permute(1, 2, 0)
            img = img.reshape(-1, 256)
            img = self.linear_proj(img)
            img = img.unsqueeze(0)
            img = self.encoder_s(img)
            img = self.linear_proj2(img)
            img = img.squeeze(0)
            r = int(np.sqrt(img.shape[0]))
            img = img.reshape(r, r, 256)
            img = img.permute(2, 0, 1)
            out = img.unsqueeze(0) # [1, 256, 2, 2]
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        
        
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayerWithSAAM(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, reduction=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Lightweight Style-Aware Attention Module (SAAM)
        self.style_conv_dw = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model, bias=False)
        self.style_conv_pw = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, bias=False)
        self.style_bn = nn.BatchNorm2d(d_model)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.semantic_fc1 = nn.Linear(d_model, d_model // reduction, bias=False)
        self.semantic_fc2 = nn.Linear(d_model // reduction, d_model, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def style_attention(self, src):
        """
        Extract style attention, focusing on visually similar regions.

        Parameters:
            src (Tensor): Input features of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: Features with enhanced style attention of shape (batch_size, seq_length, d_model).
        """
        #print("1,src is",src.shape)#[1, 1024, 512]
        # Ensure input dimensions
        batch_size, seq_length, d_model = src.size()
        spatial_dim = int(seq_length ** 0.5)  # Ensure the sequence length can form a square
        assert spatial_dim * spatial_dim == seq_length, "Sequence length must be a perfect square for 2D reshaping."

        # Reshape to 2D spatial format for convolutions
        src_2d = src.view(batch_size, d_model, spatial_dim, spatial_dim)
        #print("2,src_2d is",src_2d.shape) #[1, 512, 32, 32]
        # Style attention (local context extraction using depthwise convolution)
        
        style_attention = self.style_conv_dw(src_2d)  # Depthwise convolution for local features
        #print("3,style_attention is",style_attention.shape) #  [1, 512, 32, 32]
        
        style_attention = self.style_conv_pw(style_attention)  # Pointwise convolution for channel interaction
        #print("4,style_attention is",style_attention.shape) # [1, 512, 32, 32]
        
        style_attention = self.style_bn(style_attention)  # Batch normalization for stability
        #print("5,style_attention is",style_attention.shape) # [1, 512, 32, 32]
        # Regional style association (global spatial pooling for style consistency)
        pooled_style = F.adaptive_avg_pool2d(style_attention, (1, 1))  # Shape: (batch_size, d_model, 1, 1)
        #print("6,pooled_style is",pooled_style.shape) # [1, 512, 1, 1]
        
        pooled_style = pooled_style.expand_as(style_attention)  # Expand pooled style for broadcasting
        #print("7,pooled_style is",pooled_style.shape) #[1, 512, 32, 32]
        # Combine local and global style features
        combined_attention = style_attention * self.sigmoid(pooled_style)
        #print("8,combined_attention is",combined_attention.shape) #[1, 512, 32, 32]
        # Apply attention to the original features
        out = src_2d * combined_attention  # Apply style attention to the input
        #print("9,out is",out.shape) #[1, 512, 32, 32]
        #print("10,out.view is",out.view(batch_size, seq_length, d_model))
        return out.view(batch_size, seq_length, d_model)  # Reshape back to original dimensions

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        # Multihead Attention Branch
        src_attn = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

        # Style Attention Module (SAAM) Branch
        src_style = self.style_attention(src)

        # Combine the outputs from both branches
        src = src_attn + src_style  # Use addition for fusion; other methods like concatenation could be applied
        src = self.norm1(src)

        # Feedforward Network (FFN)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        Transformer Encoder Layer with Parallel Style-Aware Attention Module (SAAM).

        Parameters:
            src (Tensor): Input features of shape (batch_size, seq_length, d_model).
            src_mask (Optional[Tensor]): Mask for self-attention.
            src_key_padding_mask (Optional[Tensor]): Key padding mask for self-attention.
            pos (Optional[Tensor]): Positional encoding for input features.

        Returns:
            Tensor: Output features after parallel self-attention and style attention.
        """
        # Step 1: Normalize the input for stability
        src_norm = self.norm1(src)

        # Step 2: Multihead Self-Attention (MHA)
        q = k = self.with_pos_embed(src_norm, pos)  # Add positional embedding if provided
        mha_output = self.self_attn(q, k, value=src_norm, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

        # Step 3: Style-Aware Attention Module (SAAM)
        saam_output = self.style_attention(src_norm)  # Extract style-specific features

        # Step 4: Combine outputs from MHA and SAAM
        combined_output = mha_output + saam_output  # Fuse the results (additive fusion)

        # Step 5: Residual connection and normalization
        src = src + self.dropout1(combined_output)
        src = self.norm2(src)

        # Step 6: Feed-Forward Network (FFN)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model embedding dim
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

       
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory 
 
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
    
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # print (tgt.shape) # [1024, 1, 512]

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
