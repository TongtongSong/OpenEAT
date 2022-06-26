# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
# Copyright 2021 songtongmail@163.com(Tongtong Song)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from openeat.modules.embedding import PositionalEncoding
from openeat.modules.encoder import TransformerEncoder
from openeat.modules.label_smoothing_loss import LabelSmoothingLoss

from openeat.utils.common import (IGNORE_ID, add_sos_eos, log_add, th_accuracy)
from openeat.utils.mask import make_pad_mask, subsequent_mask


class LanguageModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder_num_blocks,
        d_model: int=256,
        attention_heads: int = 4,
        linear_units: int = 1024,
        dropout_rate: float=0.1,
        lsm_weight: float=0.1,
        length_normalized_loss: bool=False,
        ignore_id: int=IGNORE_ID,
        autoregressive: bool=True
    ):
        super().__init__()
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.autoregressive = autoregressive
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model,
                                        dropout_rate,
                                        attention_heads,
                                        linear_units,
                                        encoder_num_blocks)
        self.after_norm = torch.nn.LayerNorm(d_model, eps=1e-12)
        self.proj_layer = torch.nn.Linear(d_model,vocab_size)
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=self.ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss
        )
    def forward(
        self,
        input_targets: torch.Tensor,
        output_targets: torch.Tensor,
        targets_length: torch.Tensor
    ):
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            features: (Batch, Length, ...)
            features_length: (Batch, )
            targets: (Batch, Length)
            targets_length: (Batch,)
        """
        assert targets_length.dim() == 1, targets_length.shape
        # Check that batch_size is unified
        assert (input_targets.shape[0] == targets_length.shape[0]), ( input_targets.shape, targets_length.shape)
        if self.autoregressive:
            ys_in_pad, ys_out_pad = add_sos_eos(input_targets, self.sos, self.eos, self.ignore_id)
            ys_in_lens = targets_length + 1
        else:
            pad_mask = input_targets == self.ignore_id
            ys_in_pad = input_targets.masked_fill_(pad_mask,self.eos)
            ys_in_pad, ys_out_pad = input_targets, output_targets
            ys_in_lens = targets_length
        ys_in_pad = ys_in_pad.long()
        ys_out_pad = ys_out_pad.long()
        encoder_out = self._forward_encoder(ys_in_pad, ys_in_lens)
        loss = self.criterion_att(encoder_out, ys_out_pad)
        acc = th_accuracy(
            encoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss, acc

    def _forward_encoder(
        self,
        targets:torch.Tensor,
        targets_length: torch.Tensor
    ):
        tgt_mask = (~make_pad_mask(targets_length).unsqueeze(1)).to(targets.device)
        if self.autoregressive:
            # m: (1, L, L)
            m = subsequent_mask(tgt_mask.size(-1),
                                device=tgt_mask.device).unsqueeze(0)
            # tgt_mask: (B, L, L)
            tgt_mask = tgt_mask & m
        xs = self.embedding(targets)
        xs, pos_emb = self.pos_encoding(xs)
        encoder_out, _ = self.encoder(xs, tgt_mask, pos_emb)
        encoder_out = self.after_norm(encoder_out)
        encoder_out = self.proj_layer(encoder_out)
        return encoder_out


