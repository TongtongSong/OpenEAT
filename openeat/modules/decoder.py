# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Decoder definition."""
from typing import Tuple, List, Optional

import torch
from typeguard import check_argument_types

from openeat.modules.attention import MultiHeadedAttention
from openeat.modules.decoder_layer import DecoderLayer
from openeat.modules.embedding import PositionalEncoding
from openeat.modules.positionwise_feed_forward import PositionwiseFeedForward
from openeat.utils.mask import subsequent_mask, make_pad_mask
from openeat.modules.adapter import Adapter
from openeat.modules.label_smoothing_loss import LabelSmoothingLoss

class Decoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_block: int=0,
        decoder_use_adapter: bool =False,
        down_size: int=64,
        scalar: float = 0.1,
        share_embedding: bool = False,
    ):

        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        adapter_layer_args = (encoder_output_size, dropout_rate, 
                             down_size, scalar)
        adapter_layer = Adapter

        self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim)
        )
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                adapter_layer(*adapter_layer_args) if decoder_use_adapter else None,
                dropout_rate
            ) for _ in range(num_blocks)
        ])
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        if share_embedding:
            self.output_layer.weight = self.embedding.weight
    
    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: Optional[torch.Tensor]=None,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, ys_in_pad.size(1)).unsqueeze(1)
        tgt_mask = tgt_mask.to(ys_in_pad.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        tgt = ys_in_pad
        x, _ = self.embed(tgt)
        for idx, layer in enumerate(self.decoders):
            x = layer(x, tgt_mask, memory, memory_mask)
        x = self.after_norm(x)
        pre_x = x
        x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, olens, pre_x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x= decoder(x, tgt_mask, memory,memory_mask, cache=c)
            new_cache.append(x)

        y = self.after_norm(x[:, -1])
        pre_y = y
        y = self.output_layer(y)
        return y, new_cache, pre_y
        
class BiDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        dropout_rate: dropout rate
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        adapter: bool =False,
        down_size: int=64,
        scalar: float = 0.1,
    ):

        assert check_argument_types()
        super().__init__()
        self.r_num_blocks = r_num_blocks
        
        self.left_decoder = Decoder(
            vocab_size, encoder_output_size, dropout_rate,
            attention_heads, linear_units, num_blocks,r_num_blocks,
            adapter,down_size,scalar)
        if r_num_blocks > 0:
            self.right_decoder = Decoder(
                vocab_size,  encoder_output_size, dropout_rate,
                attention_heads, linear_units, r_num_blocks, num_blocks,
                adapter,down_size,scalar)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, olens, pre_l_x = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if self.r_num_blocks:
            r_x, olens, _ = self.right_decoder(memory, memory_mask, 
                             r_ys_in_pad, ys_in_lens)
        return l_x, r_x, pre_l_x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)