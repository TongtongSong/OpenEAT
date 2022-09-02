# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
# Copyright (c) 2021 songtongmail@163.com (Tongtong Song)
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
import argparse
import codecs
import logging
import random
import numpy as np

import kaldi_io

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import torchaudio
torchaudio.set_audio_backend("sox_io")
import torchaudio.compliance.kaldi as kaldi

from openeat.dataset.audio_processor import _speed_generator, _speed_perturb
from openeat.dataset.feature_processor import (_normalization, 
                                     _spec_augmentation, _spec_substitute)

from openeat.dataset.text_processor import _remove_punctuation, _tokenizer

from openeat.utils.common import IGNORE_ID

def _extract_feature(batch, feature_extraction_conf):
    """ Extract acoustic fbank feature from origin waveform.
    Args:
        batch: a list of tuple (wav id , wave path).
        feature_extraction_conf:a dict , the config of fbank extraction.
    Returns:
        (keys, feats, labels)
    """
    speed_perturb_rate = feature_extraction_conf.get('speed_perturb_rate', 0.5)
    speeds = feature_extraction_conf.get('speeds', None)
    keys = []
    feats = []
    lengths = []
    labels = []
    for i, x in enumerate(batch):
        try:
            wav = x[1]
            value = wav.strip().split(",")
            # 1 for general wav.scp, 3 for segmented wav.scp
            assert len(value) == 1 or len(value) == 3
            wav_path = value[0]      
            # value length 3 means using segmented wav.scp
            # incluede .wav, start time, end time
            sample_rate = torchaudio.backend.sox_io_backend.info(
                wav_path).sample_rate
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_path)
            #waveform = waveform.float()
            #print('1',waveform)
            waveform = waveform * (1 << 15)
            #print('2',waveform)
            if 'resample_rate' in feature_extraction_conf:
                resample_rate = feature_extraction_conf['resample_rate']
            else:
                resample_rate = sample_rate
            if resample_rate != sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)
                sample_rate = resample_rate
            
            # speed perturb

            speed = x[3]
            if random.random() < speed_perturb_rate:
                speed = _speed_generator(speeds)
            if speed != 1.0:
                waveform = _speed_perturb(waveform,sample_rate,speed)

            mat = kaldi.fbank(
                waveform,
                num_mel_bins=feature_extraction_conf['mel_bins'],
                frame_length=25,
                frame_shift=10,
                dither=feature_extraction_conf['wav_dither'],
                energy_floor=0.0,
                sample_frequency=sample_rate)

            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
            feature_length = mat.shape[0]
            labels.append(np.array(x[2]))
        except (Exception) as e:
            print(e)
            logging.warn('read utterance {} error'.format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    
    print(labels)
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels

def _load_feature(batch):
    """ Load acoustic feature from files.
    The features have been prepared in previous step, usualy by Kaldi.
    Args:
        batch: a list of tuple (wav id , feature ark path, language   ).
    Returns:
        (keys, feats, labels, languages)
    """
    
    keys = []
    feats = []
    lengths = []
    labels = []
    
    for i, x in enumerate(batch):
        try:
            keys.append(x[0])
            mat = kaldi_io.read_mat(x[1])
            feats.append(mat)
            lengths.append(mat.shape[0])
            labels.append(np.array(x[2]))
            feature_length = mat.shape[0]
            labels.append(np.array(x[2]))
        except (Exception):
            logging.warn('read utterance {} error'.format(x[0]))
            pass
    
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels


class audio_collate_func(object):
    """ Collate function for AudioDataset
    """
    def __init__(
        self,
        feature_dither=0.0,
        spec_aug=False,
        spec_aug_conf=None,
        spec_sub=False,
        spec_sub_conf=None,
        raw_wav=True,
        feature_extraction_conf=None,
        normalization=True
    ):
        """
        Args:
            raw_wav:
                    True if input is raw wav and feature extraction is needed.
                    False if input is extracted feature
        """
        self.feature_dither = feature_dither
        self.spec_sub = spec_sub
        self.spec_aug = spec_aug
        self.spec_sub_conf = spec_sub_conf
        self.spec_aug_conf = spec_aug_conf
        self.raw_wav = raw_wav
        self.feature_extraction_conf = feature_extraction_conf
        self.normalization = normalization


    def __call__(self, batch):
        if len(batch) == 1:
            batch = batch[0]
        if self.raw_wav:
            keys, xs, ys = _extract_feature(batch,self.feature_extraction_conf)
        else:
            keys, xs, ys = _load_feature(batch)
        train_flag = True
        if ys is None:
            train_flag = False
        if self.normalization:
            xs = [_normalization(x) for x in xs]
        # optional feature dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]
        
        # optinoal spec substitute
        if self.spec_sub:
            xs = [_spec_substitute(x,**self.spec_sub_conf) for i,x in enumerate(xs)]

        # optinoal spec augmentation
        if self.spec_aug:
            xs = [_spec_augmentation(x,**self.spec_aug_conf) for i,x in enumerate(xs)]

        # padding
        features_length =torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))

        # pad_sequence will FAIL in case xs is empty
        if len(xs) > 0:
            features = pad_sequence([torch.from_numpy(x).float() for x in xs],
                                  True, 0)
        else:
            features = torch.Tensor(xs)
        if train_flag:
            targets_length = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32))
            if len(ys) > 0:
                targets = pad_sequence([torch.from_numpy(y).int() for y in ys],
                                      True, IGNORE_ID)
            else:
                targets = torch.Tensor(ys)
        else:
            targets = None
            targets_length = None
        
        inputs = {
            'features': features,
            'features_length': features_length,
            'targets': targets,
            'targets_length': targets_length
        }
        return keys,  inputs

class AudioDataset(Dataset):
    def __init__(self,
                 data_file,
                 char_dict,
                 bpe_model = None,
                 max_length = 10240,
                 min_length = 0,
                 token_max_length = 200,
                 token_min_length = 0,
                 batch_type = 'static',
                 batch_size = 1,
                 max_frames_in_batch = 0,
                 sort = False,
                 speed_perturb = False,
                 speeds = [0.9, 1.1, 0.1],
                 raw_wav = True):
        """Dataset for loading audio data.
        Attributes:
            data_file: input data file
                Plain text data file, each line contains following 4 fields,
                which is split by '\t':
                    utt: utt1
                    feat: tmp/data/file1.wav or feat:tmp/data/fbank.ark:30
                    feat_shape: 4.95(in seconds) or feat_shape: 495,80(495 is in frames)
                    text: i love you
            char_dict: characters dict for changing text to token
            bpe_model: sentencepiece based bpe model for spliting English word to bpe
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length (10ms)
            token_max_length: drop utterance which is greater than token_max_length,
                especially when use char unit for english modeling
            token_min_length: drop utterance which is less than token_max_length
            batch_type: static or dynamic, see max_frames_in_batch(dynamic)
            batch_size: number of utterances in a batch,
               it's for static batch size.
            max_frames_in_batch: max feature frames in a batch,
               when batch_type is dynamic, it's for dynamic batch size.
               Then batch_size is ignored, we will keep filling the
               batch until the total frames in batch up to max_frames_in_batch.
            sort: whether to sort all data, so the utterance with the same
               length could be filled in a same batch.
            raw_wav: use raw wave or extracted featute.
                if raw wave is used, dynamic waveform-level augmentation could be used
                and the feature is extracted by torchaudio.
                if extracted featute(e.g. by kaldi) is used, only feature-level
                augmentation such as specaug could be used.
        """
        assert batch_type in ['static', 'dynamic','shuffle']
        if bpe_model is not None:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(bpe_model)
        else:
            sp = None
        self.batch_size = 1 if batch_type in ['static', 'dynamic'] else batch_size
        self.char_dict = char_dict
        self.vocab_size = len(char_dict)
        if speed_perturb:
            speed_list = [float(s) for s in np.arange(speeds[0],speeds[1],speeds[2])]
        else:
            speed_list = [1.0]
        data = []
        # Open in utf8 mode since meet encoding problem
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) != 4:
                    continue
                key = arr[0].split(':')[1]
                text = arr[3].split(':')[1]
                text = text.replace('<unk>','zzzzzz')
                text = _remove_punctuation(text)
                text = text.replace('zzzzzz','#')
                tokens = _tokenizer(text, sp)
                tokenid = [char_dict[w] if w in char_dict else char_dict['<unk>'] for w in tokens]
                if raw_wav:
                    path = ':'.join(arr[1].split(':')[1:])
                    num_frames = int(float(arr[2].split(':')[1]) * 1000 / 10)
                else:
                    path = ':'.join(arr[1].split(':')[1:])
                    feat_info = arr[2].split(':')[1].split(',')
                    feat_dim = int(feat_info[1].strip())
                    num_frames = int(feat_info[0].strip())
                    self.input_size = feat_dim
                length = num_frames
                token_length = len(tokenid)
                if min_length < length < max_length and token_min_length < token_length < token_max_length:
                    for speed in speed_list:
                        num_frames *= speed 
                        data.append((key, path, num_frames, tokenid, speed))
        if sort:
            data = sorted(data, key=lambda x: x[2])
        print(data_file)
        num_data = len(data)
        # Dynamic batch size
        if batch_type == 'dynamic':
            assert (max_frames_in_batch > 0)
            self.data = []
            self.data.append([])
            num_frames_in_batch = 0
            for i in range(num_data):
                length = data[i][2]
                num_frames_in_batch += length
                if num_frames_in_batch > max_frames_in_batch:
                    self.data.append([])
                    num_frames_in_batch = length
                self.data[-1].append((data[i][0], data[i][1], data[i][3], data[i][4]))
        
        # Static batch size
        elif batch_type == 'static':
            cur = 0
            self.data = []
            while cur < num_data:
                end = min(cur + batch_size, num_data)
                item = []
                for i in range(cur, end):
                    item.append((data[i][0], data[i][1], data[i][3], data[i][4]))
                self.data.append(item)
                cur = end
        else:
            self.data = []
            for i in range(num_data):
                self.data.append([data[i][0],data[i][1],data[i][3],data[i][4]])

        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class text_collate_func(object):
    """ Collate function for AudioDataset
    """
    def __init__(
        self,
        char_dict,
        autoregressive: bool = True
    ):
        """
        Args:
            raw_wav:
                    True if input is raw wav and feature extraction is needed.
                    False if input is extracted feature
        """
        self.char_dict = char_dict
        self.autoregressive = autoregressive

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[2])
        keys = [data[0] for data in batch]
        if self.autoregressive:
            input_targets = [data[1] for data in batch]
            output_targets = input_targets
        else:
            input_targets, output_targets = [], []
            for data in batch:
                input_target, output_target = self.random_word(data[1])
                input_targets.append(input_target)
                output_targets.append(output_target)
        targets_length = [data[2] for data in batch]
        if len(keys) > 0:
            input_targets = pad_sequence([torch.Tensor(y).int() for y in input_targets],
                                True, IGNORE_ID)
            output_targets = pad_sequence([torch.Tensor(y).int() for y in output_targets],
                                True, IGNORE_ID)
        else:
            input_targets = torch.Tensor(input_targets)
            output_targets = torch.Tensor(output_targets)
        
        targets_length = torch.Tensor(targets_length).int()
        inputs = {
            'input_targets': input_targets,
            'output_targets' : output_targets,
            'targets_length': targets_length
        }
        return keys, inputs
    
    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.char_dict['<unk>']
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(2,len(self.char_dict)-1)
                # 10% randomly change token to current token
                else:
                    tokens[i] = token
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(IGNORE_ID)

        return tokens, output_label
    
class TextDataset(Dataset):
    def __init__(
        self,
        text_file,
        char_dict,
        bpe_model,
        batch_size,
        max_length=200,
        min_length=1,
        batch_type='static',
        max_tokens_in_batch=0,
        sort=True
    ):
        """Dataset for loading audio data.
        Attributes::
            data_file: input data file
                Plain text data file, each line contains following 7 fields,
            max_length: drop utterance which is greater than token_max_length,
                especially when use char unit for english modeling
            min_length: drop utterance which is less than token_max_length
        """
        self.batch_size= batch_size
        self.input_size = -1
        if bpe_model is not None:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(bpe_model)
        else:
            sp = None
        self.vocab_size = len(char_dict)

        self.data = []
        with codecs.open(text_file,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                key = line[0]
                text = _remove_punctuation(' '.join(line[1:]))
                tokens = _tokenizer(text, sp, char_dict)
                target = [char_dict[char] if char in char_dict.keys() else 1 for char in tokens]
                target_length = len(target)
                if target_length > min_length and target_length < max_length:
                    self.data.append([key, target, target_length])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

