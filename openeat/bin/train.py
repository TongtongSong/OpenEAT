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
import torch.optim as optim
from torch.utils.data import DataLoader

from openeat.dataset.dataset import AudioDataset, audio_collate_func

from openeat.utils.checkpoint import load_trained_modules, save_checkpoint
from openeat.utils.executor import Executor
from openeat.utils.scheduler import WarmupLR
from openeat.utils.common import init_logger

from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == '__main__':
    torch.manual_seed(777)
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True,
                        help='config file')
    parser.add_argument('--model',
                        help='wenet or openeat',
                        default='openeat')
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
    parser.add_argument("--init_mods",
                    default="encoder.,ctc.,decoder.",
                    type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                    help="List of encoder modules \
                    to initialize ,separated by a comma")
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
    
    # train dataset and dataloader
    raw_wav = configs['dataset_conf']['raw_wav']
    dataset_conf = configs['dataset_conf']
    train_dataset = AudioDataset(args.train_data,
                                char_dict,
                                args.bpe_model,
                                **dataset_conf)
    collate_conf = configs['collate_conf']
    train_collate_func = audio_collate_func(**collate_conf,
                                    raw_wav=raw_wav)
    train_data_loader = DataLoaderX(train_dataset,
                                collate_fn=train_collate_func,
                                sampler=None,
                                shuffle=True,
                                batch_size=train_dataset.batch_size,
                                pin_memory=True,
                                num_workers=args.ngpus * args.num_workers)
    # cv dataset and dataloader
    if args.cv_data is not None:
        cv_dataset_conf = copy.deepcopy(dataset_conf)
        cv_dataset_conf['speed_perturb'] = False
        cv_dataset = AudioDataset(args.cv_data,
                                char_dict,
                                args.bpe_model,
                                **cv_dataset_conf)
        cv_collate_conf = copy.deepcopy(collate_conf)
        # no augmenation on cv set
        cv_collate_conf['feature_extraction_conf']['speed_perturb_rate'] = 0
        cv_collate_conf['feature_extraction_conf']['wav_dither'] = 0.0
        cv_collate_conf['feature_dither'] = 0
        cv_collate_conf['spec_sub'] = False
        cv_collate_conf['spec_aug'] = False
        cv_collate_func = audio_collate_func(**cv_collate_conf,
                                    raw_wav=raw_wav)
        cv_data_loader = DataLoaderX(cv_dataset,
                                    collate_fn=cv_collate_func,
                                    sampler=None,
                                    shuffle=False,
                                    batch_size=cv_dataset.batch_size,
                                    pin_memory=True,
                                    num_workers=args.ngpus * args.num_workers)
    # update configs
    if raw_wav:
        configs['model_conf']['input_size'] = collate_conf['feature_extraction_conf']['mel_bins']
    else:
        configs['model_conf']['input_size'] = train_dataset.input_size # -1 means lm
    configs['model_conf']['vocab_size'] = train_dataset.vocab_size

    saved_config_path = os.path.join(args.exp_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    
    if args.model == 'wenet':
        from wenet.transformer.asr_model import init_asr_model
        model = init_asr_model(configs['model_conf'])
    else:
        from openeat.models.asr_model import ASRModel
        model = ASRModel(**configs['model_conf'])
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

    step = 0
    start_epoch = 0
    if args.checkpoint is not None:
        info = load_trained_modules(model, args.checkpoint,args.init_mods)
        if info:
            step = info['step']
            start_epoch = info['epoch']+1
    
    # freeze backbone for training adapters
    encoder_adapter = configs['model_conf'].get('encoder_use_adapter', False)
    decoder_adapter = configs['model_conf'].get('decoder_use_adapter', False)
    if encoder_adapter or decoder_adapter:
        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
    
    save_model_path = os.path.join(args.exp_dir, 'init.pt')
    save_checkpoint(model, save_model_path)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, len(train_data_loader)*configs['warmup_epoch'])
    executor = Executor()
    executor.step = step
    scheduler.set_step(step)
    optimizer.param_groups[0]['lr'] = scheduler.get_lr()[0]
    max_epoch = configs.get('max_epoch', 50)
    for epoch in range(start_epoch, max_epoch):
        # Training
        logger.info('Epoch [{}/{}] lr:{:.8f}'.format(
            epoch,max_epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = executor.train(epoch, logger, model, 
                                optimizer, scheduler, train_data_loader, device, configs)
        log_str = 'Epoch [{}/{}] TRAIN Epoch Loss:{:.4f}'.format(
                epoch, max_epoch, train_loss)
        if train_acc:
                log_str += ' Acc:{:.4f}'.format(train_acc)
        logger.info(log_str)
        
        # Validation
        if args.cv_data is not None:
            cv_loss, cv_acc = executor.cv(epoch, logger,
                    model, cv_data_loader, device, configs)
            log_str = 'Epoch [{}/{}] CV Epoch Loss:{:.4f}'.format(epoch, max_epoch, cv_loss)
            if cv_acc:
                log_str += ' Acc:{:.4f}'.format(cv_acc)
            logger.info(log_str)
        else:
            cv_loss = None
            cv_acc = None

        # save checkpoint
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
