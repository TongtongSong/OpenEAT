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
import sys
import os
path = os.path.abspath(__file__)
path1 = os.path.split(path)[0]
path2 = os.path.split(path1)[0]
path3 = os.path.split(path2)[0]
sys.path.append(path3)
import argparse
import codecs
import copy
import logging
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from openeat.dataset.dataset import TextDataset, text_collate_func

from openeat.utils.checkpoint import load_checkpoint, save_checkpoint
from openeat.utils.executor import Executor
from openeat.utils.scheduler import WarmupLR
from openeat.utils.common import init_logger
from openeat.models.language_model import LanguageModel

if __name__ == '__main__':
    torch.manual_seed(777)
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True,
                        help='config file')
    parser.add_argument('--dict', required=True,
                        help='dict')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--checkpoint',
                        help='pre-trained model',
                        default=None)
    parser.add_argument('--train_data', required=True,
                        help='train data file')
    parser.add_argument('--cv_data', default=None,
                        help='cv data file')
    parser.add_argument('--exp_dir', required=True,
                        help='save model dir')
    parser.add_argument('--ngpus', type=int,
                        default=-1)
    parser.add_argument('--num_workers', type=int,
                        default=1)
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    logger = init_logger(log_file=os.path.join(args.exp_dir, 'train.log'))
    logger.info(args)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    char_dict = {}
    with codecs.open(args.dict, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            char_dict[line[0]] = int(line[1])
    
    collate_func = text_collate_func(char_dict,configs['model_conf']['autoregressive'])
    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = TextDataset(args.train_data,
                                 char_dict,
                                 **dataset_conf)
    train_data_loader = DataLoaderX(train_dataset,
                                    collate_fn=collate_func,
                                    sampler=None,
                                    shuffle=True,
                                    batch_size=train_dataset.batch_size,
                                    pin_memory=True,
                                    num_workers=args.ngpus * args.num_workers)
    if args.cv_data is not None:
        cv_dataset_conf = copy.deepcopy(dataset_conf)
        cv_dataset_conf['paste'] = False
        cv_dataset = TextDataset(args.cv_data,
                                 char_dict,
                                 **cv_dataset_conf)
        cv_data_loader = DataLoader(cv_dataset,
                                     collate_fn=collate_func,
                                     sampler=None,
                                     shuffle=False,
                                     batch_size=cv_dataset.batch_size,
                                     pin_memory=True,
                                     num_workers=args.ngpus * args.num_workers)

    configs['model_conf']['vocab_size'] = train_dataset.vocab_size
    saved_config_path = os.path.join(args.exp_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    model =LanguageModel(**configs['model_conf'])
    logger.info('{}'.format(model))

    num_params = sum(p.numel() for p in model.parameters())
    logger.info('The number of model params: {}'.format(num_params))

    if args.ngpus>1:
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.DataParallel(model,
                device_ids=[i for i in range(args.ngpus)])
        device = torch.device("cuda")
    else:
        use_cuda = args.ngpus >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                        **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    step = 0
    start_epoch = 0
    if args.checkpoint is not None:
        logger.info('Load pre-trained model from {}'.format(args.checkpoint))
        info = load_checkpoint(model, args.checkpoint)
        if info:
            step = info['step']
            start_epoch = info['epoch']+1
        
    save_model_path = os.path.join(args.exp_dir, 'init.pt')
    save_checkpoint(model, save_model_path)
    executor = Executor()
    executor.step = step
    scheduler.set_step(step)
    optimizer.param_groups[0]['lr'] = scheduler.get_lr()[0]
    max_epoch = configs.get('max_epoch', 100)
    for epoch in range(start_epoch, max_epoch):
        logger.info('Epoch [{}/{}] lr:{:.8f}'.format(
            epoch,max_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = executor.train(epoch, logger,
                model, optimizer, scheduler, train_data_loader, device, configs)
        logger.info('Epoch [{}/{}] TRAIN Epoch Loss:{:.4f} Acc:{:.4f}'.format(
            epoch, max_epoch, train_loss, train_acc))
        if args.cv_data is not None:
            cv_loss, cv_acc = executor.cv(epoch, logger,
                    model, cv_data_loader, device, configs)
            logger.info('Epoch [{}/{}] CV Epoch Loss:{:.4f} Acc:{:.4f}'.format(
                epoch, max_epoch, cv_loss, cv_acc))
        else:
            cv_loss = None

        save_model_path = os.path.join(args.exp_dir, '{}.pt'.format(epoch))
        save_checkpoint(
            model, save_model_path, {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'cv_loss': cv_loss,
                'step': executor.step
            })
