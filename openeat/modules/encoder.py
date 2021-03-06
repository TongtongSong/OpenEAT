#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
# Copyright 2021 songtongmail@163.com (Tongtong Song)

"""Encoder definition."""
from typeguard import check_argument_types

import torch

from openeat.modules.attention import RelPositionMultiHeadedAttention, MultiHeadedAttention
from openeat.modules.convolution import ConvolutionModule
from openeat.modules.encoder_layer import EncoderLayer
from openeat.modules.positionwise_feed_forward import PositionwiseFeedForward
from openeat.modules.adapter import Adapter

from openeat.utils.common import get_activation

class Encoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        dropout_rate: float = 0.1,
        attention_heads: int = 4,
        linear_units: int = 2048,
        encoder_num_blocks: int = 6,
        activation_type: str = 'swish',
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        cnn_module_norm: str = 'batch_norm',
        causal: bool = False,
        encoder_use_adapter: bool = False,
        down_size: int = 64,
        scalar: str = "0.1"
    ):
        """
        Args:
            d_model (int): dimension of attention
            dropout_rate (float): dropout rate
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed forward
            num_blocks (int): the number of decoder blocks
            adapter (bool): use Adapter or not
            down_size (int): downsampling dimension for Adapter
            scalar (float): 0-1.0, adapter weights 
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = d_model
        activation = get_activation(activation_type)
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            d_model,
            linear_units,
            dropout_rate,
            activation
        )
        if use_cnn_module: # conformer
            attention_layer = RelPositionMultiHeadedAttention
        else:
            attention_layer = MultiHeadedAttention
        attention_layer_args = (attention_heads, d_model, dropout_rate)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (d_model, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)
        adapter_layer = Adapter
        adapter_layer_args = (d_model, dropout_rate, down_size, scalar)
        self.encoders = torch.nn.ModuleList([
            EncoderLayer(
                d_model,
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                attention_layer(*attention_layer_args),
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                positionwise_layer(
                    *positionwise_layer_args),
                adapter_layer(
                    *adapter_layer_args) if encoder_use_adapter else None,
                dropout_rate
            ) for _ in range(encoder_num_blocks)
        ])
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) ->  torch.Tensor:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)

        Returns:
            encoder output tensor
        """
        for idx,layer in enumerate(self.encoders):
            xs = layer(xs, mask, pos_emb)
        return xs, mask
        