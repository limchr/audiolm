from pathlib import Path
from functools import partial, wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from audiolm_pytorch.utils import curtail_to_multiple

from einops import rearrange, reduce

from audiolm_pytorch import EncodecWrapper

import numpy as np

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def is_unique(arr):
    return len(set(arr)) == len(arr)

# dataset functions


import random
import pickle as pkl


class EncodecSoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz = 24000,
        exts = ['flac', 'wav', 'mp3', 'webm'],
        fixed_length = 2,
        device = 'cuda',
        seed = 1234
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder "{str(path)}" does not exist'
        files = sorted([str(file) for ext in exts for file in path.glob(f'**/*.{ext}')])
        random.Random(seed).shuffle(files)
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.fixed_length = fixed_length * target_sample_hz
        self.target_sample_hz = target_sample_hz
        self.device = device

        self.encodec = EncodecWrapper()
        self.encodec.to(device=device)

        # self.sound_classes = ['tom', 'snare', 'clap', 'hh', 'hihat', 'crash', 'conga', 'bdrum', 'kick', 'bd', 'perc', 'bell', 'rim', 'slap', 'tamb', 'bongo', 'cymb', 'wood', 'synth', 'clav', 'cow', 'bassdrum', 'ride', 'drum', 'bass', 'snap']

        self.class_groups = {
            'tom': ['tom', 'conga', 'bongo'],
            'kick': ['kick', 'bass', 'bassdrum', 'bd', 'bdrum'],
            'snare': ['snare', 'rim', 'snap'],
            'hihat': ['hihat', 'hh', 'cow', 'tamb', 'wood', 'ride', 'crash', 'cymb', 'bell'],
            'clap': ['clap'],
        }
        self.classes = list(self.class_groups.keys())
        
        self.fm = torch.tensor([-3.6364e-02,  1.7838e+00, -7.0549e-01, -3.1619e-01, -1.4357e-01,
         1.1365e+00, -3.8665e-01, -3.2779e-02, -5.0044e-01,  4.9229e-01,
         2.0419e-01,  6.1962e-02,  2.5571e-01, -1.0065e+00, -1.9556e-01,
        -1.0286e+00,  6.2423e-02,  4.4656e-01, -3.2442e-01,  5.2447e-02,
        -2.0215e+00, -9.5743e-01, -6.5231e-01,  4.1975e-02,  1.0135e+00,
        -7.3108e-02,  1.3141e+00, -7.9130e-01, -8.2698e-01, -4.8687e-01,
        -1.1443e+00,  6.8807e-01,  5.6460e-01, -4.0939e-01,  4.9940e-01,
        -6.4857e-01, -6.2680e-01, -1.4062e+00,  8.5884e-02,  4.2734e-01,
         1.0457e+00,  5.3954e-01,  2.0602e+00,  5.4699e-01,  2.1040e-01,
        -4.5884e-01, -7.1921e-01,  3.8900e-01,  3.7790e-01, -7.6725e-01,
        -5.5793e-01, -1.0237e+00,  2.7022e-02,  2.3536e-01, -7.0869e-01,
         1.0067e-01, -1.4789e-01, -9.0387e-01, -2.0515e-01,  9.4470e-02,
        -1.3251e+00, -8.8811e-01,  1.0066e-03, -3.4481e-01, -1.1610e+00,
        -2.3157e-01,  1.1054e-01, -5.1266e-01,  7.2692e-02,  4.9172e-01,
        -2.0810e-01, -1.0716e+00, -1.0600e-01, -4.0507e-01, -5.9553e-01,
        -2.0616e+00,  6.1707e-01, -4.2669e-01, -7.2851e-01,  1.0536e-01,
        -2.3894e-01, -1.0452e+00, -1.3781e+00,  5.8343e-01,  3.5061e-01,
        -3.6144e-01,  2.0511e-01, -2.1110e-01,  5.9495e-01, -2.7539e-01,
         2.0501e+00,  1.0595e+00,  1.0206e-01,  5.6176e-01, -1.3392e+00,
         1.2255e+00,  2.2466e-01,  2.4444e+00, -1.7396e-02,  5.1664e-01,
         4.0743e-01,  1.2404e+00,  7.5625e-01, -2.8410e-01, -1.5373e+00,
        -1.7284e-01, -2.4447e-01, -3.2267e-01, -3.6607e-01,  6.4165e-01,
        -1.7162e+00, -4.5967e-02, -1.1265e+00,  9.5024e-02, -9.0621e-01,
         1.4869e-01, -6.5438e-02, -3.8477e-01,  7.2008e-02, -5.7455e-01,
         9.6864e-02, -1.4728e-01, -1.5250e-01,  4.2296e-01, -2.9448e-01,
        -7.0547e-02, -3.8767e-01,  3.8826e-01]).to(device=device)
        
        



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        filename = file.__str__()[file.__str__().rfind('/')+1:]
        
        # define class vector
        class_vec = torch.zeros(len(self.class_groups.keys()))
        for i, (key, value) in enumerate(self.class_groups.items()):
            in_group = [1.0 if name in filename.lower() else 0.0 for name in value]
            class_vec[i] = 1.0 if sum(in_group)>0.0 else 0.0
        
        # todo: this is kinda dirty (selecting one class randomly if there are multiple)
        y_numeric = -1        
        if sum(class_vec) > 0.0: # one or more classes found
            class_indices = np.where(class_vec>0.0)[0]
            y_numeric = random.choice(class_indices)
            if (sum(class_vec)>1): 
                print('found multiple classes for name '+filename+' (choosing one randomly)')
                pass
                # class_vec[:] = 0.0
                # class_vec[rand_class] = 1.0
        
        # class_vec = torch.tensor([1.0 if name in filename.lower() else 0.0 for name in self.sound_classes])
        class_vec = class_vec.to(device=self.device)
        # if not class_vec.any(): print('couldnt find class for name: '+filename)

        # load audio from file and encode it
        wav_data = self.load_audio(file)
        x, x_transposed = self.encode_sample(wav_data)



        return x, x_transposed, class_vec, y_numeric
    
    def load_audio(self, file):
        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # first resample data to the max target freq

        data = resample(data, sample_hz, self.target_sample_hz)

        audio_length = data.size(1)
        
        # unifying the length of samples by cropping too long samples
        # of by filling too short samples with zeros
        if audio_length > self.fixed_length:
            data = data[:, 0:self.fixed_length]
        else:
            data2 = torch.zeros((1,self.fixed_length),dtype=torch.float32)
            data2[0,:data.shape[1]] = data
            data = data2

        # todo: what is this for? Important for saving the audiofile back later?
        data = data.squeeze() # squeeze first dimension with size 1
        
        data = data.to(self.device)
        
        return data

    
    def encode_sample(self,wav_data):
        codes, indices, _ = self.encodec.forward(wav_data, 24000, True)

        codes = codes / 5.0

        # subtract mean per feature dimension

        codes = codes - self.fm



        # shift data for target array construction
        # x = 0 1 2 3 4
        # y = 1 2 3 4

        x, y = codes[:-1], codes
        x, y = x.to(device=self.device), y.to(device=self.device)
        x = torch.cat((torch.zeros((1,128)).to(self.device),x),dim=0)
        # y = torch.cat((torch.zeros((1,128)).to(self.device),y),dim=0)
        return x, y

    def decode_sample(self, codes):
        codes = codes + self.fm
        
        codes = codes * 5.0

        codes = codes[:,1:,:] # remove first zero vector
        wav = self.encodec.decode(codes)
        return wav
    
    def save_audio(self, wav, filename):
        torchaudio.save(filename, wav.to('cpu').reshape(1,-1), 24000)


class BufferedDataset(Dataset):
    def __init__(self, dataset, dump_path, dump_ds = False):
        self.dataset = dataset
        self.dump_path = dump_path
        if dump_ds:
            self.dump_ds()
        else:
            self.load_ds()
    def dump_ds(self):
        ds = []
        for d in self.dataset:
            ds.append(d)
        self.ds = ds
        with open(self.dump_path, 'wb') as f:
            pkl.dump(self.ds, f)
        print('dumping dataset to '+self.dump_path+' done')
    def load_ds(self):
        with open(self.dump_path, 'rb') as f:
            self.ds = pkl.load(f)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class SoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz: Union[int, Tuple[int, ...]],  # target sample hz must be specified, or a tuple of them if one wants to return multiple resampled
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: Optional[int] = None,               # max length would apply to the highest target_sample_hz, if there are multiple
        seq_len_multiple_of: Optional[Union[int, Tuple[Optional[int], ...]]] = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder "{str(path)}" does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.max_length = max_length
        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        # strategy, if there are multiple target sample hz, would be to resample to the highest one first
        # apply the max lengths, and then resample to all the others

        self.max_target_sample_hz = max(self.target_sample_hz)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # first resample data to the max target freq

        data = resample(data, sample_hz, self.max_target_sample_hz)
        sample_hz = self.max_target_sample_hz

        # then curtail or pad the audio depending on the max length

        max_length = self.max_length
        audio_length = data.size(1)

        if exists(max_length):
            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]
            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

        data = rearrange(data, '1 ... -> ...')

        # resample if target_sample_hz is not None in the tuple

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        data_tuple = tuple(resample(d, sample_hz, target_sample_hz) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually for curtailing to multiple

        for data, seq_len_multiple_of in zip(data_tuple, self.seq_len_multiple_of):
            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
