#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
# Copyright 2022 songtongmail@163.com (Tongtong Song)

"""Encoder self-attention layer definition."""

from typing import Optional

import torch
from torch import nn

class EncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        Adapter (Optional[torch.nn.Module]): Adapter module instance.
            `Adapter` or `None` .
        dropout_rate (float): Dropout rate.
    """
    def __init__(
        self,
        size: int,
        feed_forward_macaron: Optional[torch.nn.Module],
        self_attn: torch.nn.Module,
        conv_module: Optional[torch.nn.Module],
        feed_forward: torch.nn.Module,
        adapter: Optional[torch.nn.Module],
        dropout_rate: float = 0.1,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.size = size
        self.feed_forward_macaron = feed_forward_macaron
        self.self_attn = self_attn
        self.conv_module = conv_module
        self.feed_forward = feed_forward
        self.adapter = adapter
        self.ff_scale = 1
        if feed_forward_macaron:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if conv_module:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # for the CNN module
            
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_final = nn.LayerNorm(size, eps=1e-12)  # for the final output of the block
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute encoded features.
        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, 1，time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
        """
        if self.feed_forward_macaron: # conformer
            residual = x
            x = self.norm_ff_macaron(x)
            x_ffm = self.feed_forward_macaron(x)
            x = residual + self.ff_scale * self.dropout(x_ffm)

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, mask, pos_emb)
        x = residual + self.dropout(x_att)

        if self.conv_module: # conformer
            residual = x
            x = self.norm_conv(x)
            x = self.conv_module(x, mask)
            x = residual + self.dropout(x)

        if self.adapter:
            adapt_x = self.adapter(x)
        
        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x_ff = self.feed_forward(x)
        x = residual + self.ff_scale * self.dropout(x_ff)

        if self.adapter:
            x = x + adapt_x
        x = self.norm_final(x)
        return x
