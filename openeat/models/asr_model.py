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
from openeat.utils.cmvn import load_cmvn
from openeat.modules.cmvn import GlobalCMVN
from openeat.modules.encoder import TransformerEncoder
from openeat.modules.decoder import BiDecoder
from openeat.modules.ctc import CTC
from openeat.modules.label_smoothing_loss import LabelSmoothingLoss


from openeat.utils.common import (IGNORE_ID, reverse_pad_list, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy)

from openeat.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        encoder_num_blocks: int,
        decoder_num_blocks: int,
        r_decoder_num_blocks: int,
        is_json_cmvn: bool=False,
        cmvn_file: str=None,
        input_layer: str='conv2d',
        pos_enc_layer_type: str='abs_pos',
        d_model: int=256,
        attention_heads: int = 4,
        linear_units: int = 1024,
        dropout_rate: float = 0.1,
        activation_type: str = 'swish',
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        casual: bool=False,
        encoder_use_adapter: bool = False,
        decoder_use_adapter: bool = False,
        down_size: int = 64,
        scalar: float = 0.1,
        ctc_weight: float = 0.3,
        lsm_weight: float = 0.1,
        reverse_weight: float = 0.0,
        length_normalized_loss: bool = False,
        ignore_id = IGNORE_ID
    ):
        super().__init__()
        self.input_size=input_size
        self.vocab_size = vocab_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        if cmvn_file is not None:
            mean, istd = load_cmvn(cmvn_file, is_json_cmvn)
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float())
        else:
            global_cmvn = None
        encoder_args = (input_size, input_layer, pos_enc_layer_type,
                        d_model, dropout_rate, attention_heads, linear_units, 
                        activation_type, 
                        macaron_style, use_cnn_module, cnn_module_kernel, casual,
                        encoder_use_adapter, down_size, scalar)
        self.encoder = TransformerEncoder(*encoder_args, 
                        global_cmvn=global_cmvn, encoder_num_blocks=encoder_num_blocks)

        self.ctc = CTC(vocab_size, d_model, length_normalized_loss)

        # decoder
        decoder_args = (vocab_size,
                        d_model, dropout_rate, attention_heads, linear_units, 
                        decoder_num_blocks, r_decoder_num_blocks,
                        decoder_use_adapter, down_size, scalar)
        self.decoder = BiDecoder(*decoder_args)

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss
        )
    
    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        targets: torch.Tensor,
        targets_length: torch.Tensor,
    )-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            features: (Batch, Length, ...)
            features_length: (Batch, )
            targets: (Batch, Length)
            targets_length: (Batch,)
        """
        assert targets_length.dim() == 1, targets_length.shape
        # Check that batch_size is unified
        assert (features.shape[0] == features_length.shape[0] == targets.shape[0] ==
                targets_length.shape[0]), (features.shape, features_length.shape,
                                         targets.shape, targets_length.shape)
        encoder_out, encoder_mask = self.encoder(features, features_length)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, targets, targets_length)
        if self.ctc_weight < 1:
            loss_att, acc_att = \
                self._calc_att_loss(encoder_out, encoder_mask, targets, targets_length)
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            acc = acc_att
        else:
            loss = loss_ctc
            acc = None
        return loss, acc
    
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        r_ys_in_pad = torch.tensor(0.0)
        if self.reverse_weight > 0:
            # reverse the seq, used for right to left decoder
            r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
            r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                    self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)

        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)

        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att,acc_att

    def recognize(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        beam_size: int = 10
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder
        Args:
            features (torch.Tensor): (batch, max_len, feat_dim)
            features_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert features.shape[0] == features_length.shape[0]
        device = features.device
        batch_size = features.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(features, features_length)
        #encoder_out = self.affine_norm(self.affine_linear(encoder_out))
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache= None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            p, cache, _ = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache=cache)
            logp = torch.nn.functional.log_softmax(p,dim=-1)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)
        
        scores = scores.view(batch_size, beam_size)
        # 3. Select best of best
        best_index = torch.argmax(scores, dim=-1).long()
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps

    def ctc_greedy_search(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            features (torch.Tensor): (batch, max_len, feat_dim)
            features_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
        Returns:
            List[List[int]]: best path result
        """
        assert features.shape[0] == features_length.shape[0]
        batch_size = features.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self.encoder(features, features_length)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps

    def _ctc_prefix_beam_search(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        beam_size: int
    ) -> Tuple[List[List[int]],torch.Tensor]:
        """ CTC prefix beam search inner implementation
        Args:
            features (torch.Tensor): (batch, max_len, feat_dim)
            features_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert features.shape[0] == features_length.shape[0]
        batch_size = features.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self.encoder(features, features_length)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        beam_size: int
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            features (torch.Tensor): (batch, max_len, feat_dim)
            features_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(features, features_length,
                                               beam_size)
        return hyps[0][0]

    def attention_rescoring(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        beam_size: int,
        ctc_weight: float = 0.0,
        reverse_weight: float = 0.0,
        lm: Optional[torch.nn.Module]=None,
        lm_weight: float=0,
        autoregressive: bool = True,
        token2char: dict = {}
    ) -> Tuple[List[int],torch.Tensor,torch.Tensor]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out
        Args:
            features (torch.Tensor): (batch, max_len, feat_dim)
            features_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
        Returns:
            List[int]: Attention rescoring result
        """
        assert features.shape[0] == features_length.shape[0]
        device = features.device
        batch_size = features.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            features, features_length, beam_size)
        # encoder_out = self.affine_norm(self.affine_linear(encoder_out))
        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)

        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, pre_decoder_out = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)

        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()

        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()

        if lm_weight>0 and isinstance(lm,torch.nn.Module):
            if autoregressive:
                lm_input = hyps_pad
                lm_input_length = hyps_lens
            else:
                pad_mask = ori_hyps_pad == self.ignore_id
                lm_input = ori_hyps_pad.masked_fill_(pad_mask, self.eos)
                lm_input_length = hyps_lens-1
            lm_output = lm.encoder(lm_input, lm_input_length)
            lm_output = F.log_softmax(lm_output,dim=-1)
            lm_output = lm_output.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            content = []
            lm_score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
                if lm_weight>0 and isinstance(lm,torch.nn.Module):
                    lm_score += lm_output[i][j][w]
                content.append(token2char[w])
            score += decoder_out[i][len(hyp[0])][self.eos]

            if lm_weight > 0 and not isinstance(lm,torch.nn.Module):
                lm_score = lm.score(' '.join(content), bos=True, eos=True) 

            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight

            # add ctc score
            score += hyp[1] * ctc_weight 
            score += lm_score * lm_weight 

            if score > best_score:
                best_score = score
                best_index = i
        
        return hyps[best_index][0], encoder_out, pre_decoder_out
