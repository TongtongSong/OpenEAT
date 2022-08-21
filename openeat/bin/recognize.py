# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Copyright 2021 songtongmail@163.com (Tongtong Song)

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

from __future__ import print_function

import os
import sys
path = os.path.abspath(__file__)
path1 = os.path.split(path)[0]
path2 = os.path.split(path1)[0]
path3 = os.path.split(path2)[0]
sys.path.append(path3)
import argparse
import copy
import codecs

import torch
import yaml
from torch.utils.data import DataLoader

from openeat.dataset.dataset import AudioDataset, audio_collate_func
from openeat.utils.checkpoint import load_checkpoint
from openeat.utils.common import init_logger, map_to_device
from openeat.models.asr_model import ASRModel

from openeat.models.language_model import LanguageModel
import kenlm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config',
                        required=True,
                        help='config file')
    parser.add_argument('--test_data',
                        required=True,
                        help='test data file')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--raw_wav', type=bool,
                        default=True,
                        help='whether raw wav')
    parser.add_argument('--num_workers', type=int,
                        default=1)
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint',
                        required=True,
                        help='checkpoint model')
    parser.add_argument('--lm', type=str,
                        default=None,
                        help='language model')
    parser.add_argument('--lm_config', type=str,
                        default=None,
                        help='language model config')
    parser.add_argument('--lm_weight',type=float,
                        default=0.1,
                        help='language model weight for rescoring')
    parser.add_argument('--dict',
                        required=True,
                        help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--result_file',
                        required=True,
                        help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=['ctc_greedy_search','ctc_prefix_beam_search','attention', 'attention_rescoring'],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='reverse weight for reverse decoder in decoding')
    args = parser.parse_args()
    logger = init_logger(log_file=os.path.join(os.path.dirname(args.result_file), 'recognize.log'))
    logger.info(args)
    if args.mode=='attention_rescoring' and args.batch_size > 1:
        logger.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            char_dict[arr[0]] = int(arr[1])
    token2char = {v:k for k,v in char_dict.items()}
     
    configs['dataset_conf']['raw_wav'] = args.raw_wav
    configs['dataset_conf']['min_length'] = 1
    configs['dataset_conf']['max_length'] = 100000
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['speed_perturb'] = False
    dataset_conf['batch_type'] = 'static'
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['sort'] = False
    
    test_dataset = AudioDataset(args.test_data,
                                char_dict,
                                args.bpe_model,
                                **dataset_conf)
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['feature_extraction_conf']['speed_perturb_rate'] = 0
    test_collate_conf['feature_extraction_conf']['wav_dither'] = 0.0
    test_collate_conf['feature_dither'] = 0
    test_collate_conf['spec_sub'] = False
    test_collate_conf['spec_aug'] = False
    test_collate_func = audio_collate_func(**test_collate_conf,
                                    raw_wav=args.raw_wav)
    
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=1)
    
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # ASR Model
    model = ASRModel(**configs['model_conf'])
    # logger.info('{}'.format(model))
    num_params = sum(p.numel() for p in model.parameters())
    logger.info('The number of model params: {}'.format(num_params))
    eos = len(char_dict) - 1
    info = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    autoregressive = False
    if args.lm_weight>0:
        assert args.lm is not None, 'lm is None'
        if args.lm_config:
            with open(args.lm_config, 'r') as fin:
                lm_configs = yaml.load(fin, Loader=yaml.FullLoader)
                autoregressive = lm_configs['model_conf']['autoregressive']
            lm = LanguageModel(**lm_configs['model_conf'])
            num_params = sum(p.numel() for p in lm.parameters())
            logger.info('The number of languague model params: {}'.format(num_params))
            load_checkpoint(lm, args.lm)
            lm = lm.to(device)
            lm.eval()
        else:
            lm=kenlm.LanguageModel(args.lm)
    else:
        lm = None
    
    total = len(test_data_loader)
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, (keys, batch) in enumerate(test_data_loader):
            batch = map_to_device(batch,device)
            features = batch['features']
            features_length = batch['features_length']
            targets = batch['targets']
            targets_length = batch['targets_length']
            if args.mode == 'attention':
                hyps = model.recognize(
                    features,
                    features_length,
                    beam_size=args.beam_size)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'attention_rescoring':
                assert (features.size(0) == 1)
                hyps = model.attention_rescoring(
                    features,
                    features_length,
                    beam_size=args.beam_size,
                    ctc_weight = args.ctc_weight,
                    reverse_weight = args.reverse_weight,
                    lm = lm,
                    lm_weight=args.lm_weight,
                    autoregressive = autoregressive,
                    token2char = token2char
                )
                hyps = [hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps = model.ctc_greedy_search(
                    features,
                    features_length)
            elif args.mode == 'ctc_prefix_beam_search':
                assert (features.size(0) == 1)
                hyps = model.ctc_prefix_beam_search(
                    features,
                    features_length,
                    beam_size=args.beam_size)
                hyps = [hyps]
            
            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(token2char[w])
                logger.info('{}/{}:{} {}'.format(batch_idx+1, total, key, ' '.join(content)))
                fout.write('{} {}\n'.format(key,  ' '.join(content)))
