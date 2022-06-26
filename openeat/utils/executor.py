# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import torch
from torch.nn.utils import clip_grad_norm_
from openeat.utils.common import map_to_device

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, epoch,logger, model, optimizer, scheduler, data_loader, device, args):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 5.0)
        accum_grad = args.get('accum_grad', 1)
        logger.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))

        log_interval = args.get('log_interval', 10)
        num_seen_utts = 0
        total_loss = 0
        total_acc = 0
        num_total_batch = len(data_loader)
        for batch_idx, (keys, batch) in enumerate(data_loader):
            batch = map_to_device(batch,device)
            num_utts = len(keys)
            if num_utts == 0:
                continue
            loss, acc = model(**batch)
            loss = torch.mean(loss) / accum_grad
            if acc is not None:
                acc = torch.mean(acc)
            if torch.isfinite(loss):
                num_seen_utts += num_utts
                total_loss += loss.item() * accum_grad * num_utts
                if acc is not None:
                    total_acc += acc.item() * num_utts
            loss.backward()
            if batch_idx % accum_grad == 0:
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1

            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch[{}/{}] '.format(
                    batch_idx, num_total_batch)
                log_str += 'Loss:{:.4f} ALoss:{:.4f} '.format(
                    loss.item() * accum_grad, total_loss/num_seen_utts)
                if acc is not None:
                    log_str += 'Acc:{:.4f} AAcc:{:.4f} '.format(
                        acc.item(), total_acc/num_seen_utts)
                log_str += 'lr:{:.8f}'.format(lr)
                logger.info(log_str)
        return total_loss/num_seen_utts, total_acc/num_seen_utts

    def cv(self, epoch, logger, model, data_loader, device, args):
        ''' Cross validation
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        num_seen_utts = 0
        total_loss = 0
        total_acc = 0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, (keys, batch) in enumerate(data_loader):
                batch = map_to_device(batch,device)
                num_utts = len(keys)
                if num_utts == 0:
                    continue
                loss,acc = model( **batch)
                loss = torch.mean(loss)
                if acc is not None:
                    acc = torch.mean(acc)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                    if acc is not None:
                        total_acc += acc.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch[{}/{}] '.format(
                        batch_idx, num_total_batch)
                    log_str += 'Loss:{:.4f} ALoss:{:.4f} '.format(
                        loss.item(), total_loss/num_seen_utts)
                    if acc is not None:
                        log_str += 'Acc:{:.4f} AAcc:{:.4f} '.format(
                            acc.item(), total_acc/num_seen_utts)
                    logger.info(log_str)

        return total_loss/num_seen_utts, total_acc/num_seen_utts
