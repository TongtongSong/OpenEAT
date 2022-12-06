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
import yaml

import torch
import torch.distributed as dist # ddp
import torch.optim as optim
from torch.utils.data import DataLoader

from openeat.dataset.dataset import AudioDataset, audio_collate_func

from openeat.utils.checkpoint import load_trained_modules, save_checkpoint
from openeat.utils.executor import Executor
from openeat.utils.scheduler import WarmupLR
from openeat.utils.common import init_logger

from openeat.models.asr_model import ASRModel

from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == '__main__':
    torch.manual_seed(777)
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True,
                        help='config file')
    parser.add_argument('--data_type',
                        default="feat", # kaldi or wav
                        help='feat or wav')
    parser.add_argument('--train_data', required=True,
                        help='train data file')
    parser.add_argument('--cv_data', 
                        default=None,
                        help='cv data file')
    parser.add_argument('--exp_dir', required=True,
                        help='save model dir')
    parser.add_argument('--dict', required=True,
                        help='dict')
    parser.add_argument('--cmvn_file',
                        default=None,
                        help='cmvn file for global CMVN')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--checkpoint',
                        help='pre-trained model',
                        default=None)
    parser.add_argument('--gpuid',
                        default="-1",
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--num_workers', type=int,
                        default=1)
    parser.add_argument('--local_rank', type=int,
                        default=-1)
    parser.add_argument("--init_mods",
                    default="encoder.,ctc.,decoder.",
                    type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                    help="List of encoder modules \
                    to initialize ,separated by a comma")
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)

    logger = init_logger(log_file=os.path.join(args.exp_dir, 'train.log'))
    logger.info(args)
    ngpus = len(args.gpuid.split(','))
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    # os.environ['LOCAL_RANK'] = str(args.local_rank)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    char_dict = {}
    with codecs.open(args.dict, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            char_dict[line[0]] = int(line[1])
    

    # train dataset and dataloader
    dataset_conf = configs['dataset_conf']
    dataset_conf['data_type'] = args.data_type
    if args.data_type == 'feat':
        dataset_conf['speed_perturb'] = False
    train_dataset = AudioDataset(args.train_data,
                                char_dict,
                                args.bpe_model,
                                **dataset_conf)
    
    cv_dataset_conf = copy.deepcopy(dataset_conf)
    cv_dataset_conf['speed_perturb'] = False
    cv_dataset = AudioDataset(args.cv_data,
                            char_dict,
                            args.bpe_model,
                            **cv_dataset_conf)
    is_distributed = ngpus > 1
    configs['is_distributed'] = is_distributed
    if is_distributed:
        logger.info('training on multiple gpus, this gpu {}'.format(args.gpuid))
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                rank=args.local_rank,
                                world_size=ngpus)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        cv_sampler = torch.utils.data.distributed.DistributedSampler(
            cv_dataset, shuffle=False)
    else:
        train_sampler = None
        cv_sampler = None


    collate_conf = configs['collate_conf']
    
    train_collate_func = audio_collate_func(**collate_conf,
                                    data_type=args.data_type)

    cv_collate_conf = copy.deepcopy(collate_conf)
    
    # no augmenation on cv set
    cv_collate_conf['feature_extraction_conf']['speed_perturb_rate'] = 0
    cv_collate_conf['feature_extraction_conf']['wav_dither'] = 0.0
    cv_collate_conf['feature_dither'] = 0
    cv_collate_conf['spec_sub'] = False
    cv_collate_conf['spec_aug'] = False
    cv_collate_func = audio_collate_func(**cv_collate_conf,
                                data_type=args.data_type)
    train_data_loader = DataLoader(train_dataset,
                                    collate_fn=train_collate_func,
                                    sampler=train_sampler,
                                    shuffle=(train_sampler is None),
                                    batch_size=train_dataset.batch_size,
                                    pin_memory=True,
                                    num_workers=args.num_workers)

    cv_data_loader = DataLoader(cv_dataset,
                                    collate_fn=cv_collate_func,
                                    sampler=cv_sampler,
                                    shuffle=False,
                                    batch_size=cv_dataset.batch_size,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
        
    # update configs
    if args.data_type=='wav':
        configs['model_conf']['input_size'] = collate_conf['feature_extraction_conf']['mel_bins']
    else:
        configs['model_conf']['input_size'] = train_dataset.input_size # -1 means lm
    configs['model_conf']['vocab_size'] = train_dataset.vocab_size
    configs['model_conf']['cmvn_file'] = args.cmvn_file


    saved_config_path = os.path.join(args.exp_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    
    model = ASRModel(**configs['model_conf'])
    # logger.info('{}'.format(model))

    step = 0
    start_epoch = 0
    max_epoch = configs.get('max_epoch', 50)
    final_epoch = max_epoch

    # load checkpoint
    if args.checkpoint:
        info = load_trained_modules(model, args.checkpoint,args.init_mods)
        if info:
            step = info['step']
            start_epoch = info['epoch']+1
    
    # freeze backbone for training adapters
    encoder_adapter = configs['model_conf'].get('encoder_use_adapter', False)
    decoder_adapter = configs['model_conf'].get('decoder_use_adapter', False)
    if encoder_adapter or decoder_adapter:
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
    
    total_num_params = sum(p.numel() for p in model.parameters())
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    training_num_params = sum(p.numel() for p in model_parameters)
    logger.info('The number of model params: {}/{}'.format(training_num_params,total_num_params))
    if is_distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        use_cuda = ngpus >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
        local_rank = 0

    if start_epoch == 0 and local_rank == 0:
        save_model_path = os.path.join(args.exp_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, len(train_data_loader)*configs['warmup_epoch'])

    executor = Executor()
    executor.step = step
    scheduler.set_step(step)
    optimizer.param_groups[0]['lr'] = scheduler.get_lr()[0]
    
    for epoch in range(start_epoch, max_epoch):
        # Training
        if is_distributed:
            train_sampler.set_epoch(epoch)
        logger.info('Epoch [{}/{}] lr:{:.8f}'.format(
            epoch,max_epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = executor.train(logger, model, 
                                optimizer, scheduler, 
                                train_data_loader, device, configs, local_rank)
        log_str = 'Epoch [{}/{}] TRAIN Epoch Loss:{:.4f}'.format(
                epoch, max_epoch, train_loss)
        if train_acc:
            log_str += ' Acc:{:.4f}'.format(train_acc)
        log_str += ' rank:{}'.format(local_rank)
        logger.info(log_str)
        
        # Validation
        if args.cv_data:
            cv_loss, cv_acc = executor.cv(logger, model, 
                                cv_data_loader, device, configs,local_rank)
            log_str = 'Epoch [{}/{}] CV Epoch Loss:{:.4f}'.format(epoch, max_epoch, cv_loss)
            if cv_acc:
                log_str += ' Acc:{:.4f}'.format(cv_acc)
            log_str += ' rank:{}'.format(local_rank)
            logger.info(log_str)
        else:
            cv_loss = None
            cv_acc = None
        # save checkpoint
        if local_rank == 0:
            save_model_path = os.path.join(args.exp_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': train_loss,
                    'train_acc':train_acc,
                    'cv_loss': cv_loss,
                    'cv_acc': cv_acc,
                    'step': executor.step
                })
            final_epoch = epoch
    if final_epoch and local_rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
