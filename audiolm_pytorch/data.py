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

        self.sound_classes = ['tom', 'snare', 'clap', 'hh', 'hihat', 'crash', 'conga', 'bdrum', 'kick', 'bd', 'perc', 'bell', 'rim', 'slap', 'tamb', 'bongo', 'cymb', 'wood', 'synth', 'clav', 'cow', 'bassdrum', 'ride', 'drum', 'bass', 'snap']


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        filename = file.__str__()[file.__str__().rfind('/')+1:]
        
        # define class vector
        class_vec = torch.tensor([1.0 if name in filename.lower() else 0.0 for name in self.sound_classes])
        class_vec = class_vec.to(device=self.device)
        # if not class_vec.any(): print('couldnt find class for name: '+filename)

        # load audio from file and encode it
        wav_data = self.load_audio(file)
        x, y = self.encode_sample(wav_data)



        return x, y, class_vec
    
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

        codes = codes * 0.2

        # shift data for target array construction
        # x = 0 1 2 3 4
        # y = 1 2 3 4

        x, y = codes[:-1], codes
        x, y = x.to(device=self.device), y.to(device=self.device)
        x = torch.cat((torch.zeros((1,128)).to(self.device),x),dim=0)
        # y = torch.cat((torch.zeros((1,128)).to(self.device),y),dim=0)
        return x, y

    def decode_sample(self, codes):
        codes = codes * 5

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
