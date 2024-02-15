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


from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, random_split

import numpy as np

from sklearn.model_selection import train_test_split


# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def is_unique(arr):
    return len(set(arr)) == len(arr)







def get_class_weighted_sampler(ds):
    y = []
    for d in ds:
        y.append(d[2])
    y = torch.stack(y).cpu().numpy()

    class_occurrences = [len(np.where(y[:,yy] > 0.)[0]) for yy in range(y.shape[1])]
    total_occurrences = np.sum(class_occurrences)
    class_occurrences_normalized = class_occurrences / total_occurrences
    inverted_normalized_class_occurrences = 1/class_occurrences_normalized

    sample_weights = np.zeros(len(y))
    for i in range(len(y)):
        if y[i].sum() > 0: # only use samples that have at least one class (otherwise the sample weight is 0)
            sample_weights[i] = inverted_normalized_class_occurrences[y[i].argmax()] # not quite correct because there could be multiple classes

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler




def get_audio_dataset(audiofile_paths, 
                      dump_path, 
                      build_dump_from_scratch, 
                      only_labeled_samples,
                      test_size,
                      equalize_class_distribution,
                      equalize_train_data_loader_distribution,
                      batch_size,
                      seed,
                      num_samples=None
                      ):
    pds = EncodecSoundDataset(folders=audiofile_paths, length=2, device='cpu', seed=seed)
    dsb = BufferedDataset(pds, dump_path, build_dump_from_scratch)
    
    if only_labeled_samples:
        labeled_idx = [i for i in range(len(dsb)) if dsb[i][3] >= 0]
        dsb = torch.utils.data.Subset(dsb, labeled_idx)
    if not num_samples is None:
        dsb = torch.utils.data.Subset(dsb,list(range(0,num_samples)))
        
    
    stratify = [yy[3].item() for yy in dsb] if equalize_class_distribution else None
    train_indices, val_indices = train_test_split(np.arange(len(dsb)), 
                                                test_size=test_size, 
                                                random_state=seed,
                                                shuffle=True,
                                                stratify=stratify)

    ds_train = torch.utils.data.Subset(dsb, train_indices)
    ds_val = torch.utils.data.Subset(dsb, val_indices)

    if equalize_train_data_loader_distribution:
        sampler = get_class_weighted_sampler(ds_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler) # not shuffled should be ok because we shuffle in data set class
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True) # for validation we don't need a sampler, right?
    
    # make the original dataset available for some properties and functionalities (danger: no sampling or split applied)
    dsb.ds = pds
    ds_train.ds = pds
    ds_val.ds = pds
    
    return dsb, ds_train, ds_val, dl_train, dl_val



#
## some sanity debug checks
#

# # select only very few samples for debugging
# ds_train = torch.utils.data.Subset(ds_train, range(0,10))
# ds_val = ds_train

# # check if the dataset is balanced and how the class distribution is
# ys_numeric_train = [yy[3] for yy in ds_train]
# ys_numeric_test = [yy[3] for yy in ds_val]
# for i in range(-1,5):
#     ratiotrain = np.where(np.array(ys_numeric_train)==i)[0].shape[0]/len(ys_numeric_train)
#     ratiotest = np.where(np.array(ys_numeric_test)==i)[0].shape[0]/len(ys_numeric_test)
#     print('class %d train ratio: %.4f \t test ratio: %.4f' % (i, ratiotrain, ratiotest))










# ds mit nur gelabelten samples
# uebergeordnete funktion anpassen
# variable length implementieren





# dataset functions


import random
import pickle as pkl


class EncodecSoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folders,
        length, # in seconds, None for no fixed length
        device,
        seed
    ):

        # define object variables
        self.rnd = random.Random(seed)
        self.target_sample_hz = 24000
        self.length = length * self.target_sample_hz
        self.device = device
        self.seed = seed

        # extract all audio files from the folders
        audio_file_extensions = ['flac', 'wav', 'mp3', 'webm']
        all_files = []
        super().__init__()
        for folder in folders:
            path = Path(folder, follow_symlinks=True)
            assert path.exists(), f'folder "{str(path)}" does not exist'
            files = sorted([str(file) for ext in audio_file_extensions for file in path.glob(f'**/*.{ext}')])
            self.rnd.shuffle(files)
            assert len(files) > 0, 'no sound files found in folder '+ folder
            all_files.extend(files)
        self.rnd.shuffle(all_files) # shuffle everything again
        self.files = all_files

        # load encodec model
        self.encodec = EncodecWrapper()
        self.encodec.to(device=device)


        # todo: maybe make this as parameters (problem: different type of labelings in other data sets than file names)

        # self.sound_classes = ['tom', 'snare', 'clap', 'hh', 'hihat', 'crash', 'conga', 'bdrum', 'kick', 'bd', 'perc', 'bell', 'rim', 'slap', 'tamb', 'bongo', 'cymb', 'wood', 'synth', 'clav', 'cow', 'bassdrum', 'ride', 'drum', 'bass', 'snap']

        # class groups ()
        self.class_groups = {
            'tom': ['tom', 'conga', 'bongo'],
            'kick': ['kick', 'bassdrum', 'bd', 'bdrum'], # 'bass',
            'snare': ['snare', 'rim', 'snap'],
            'hihat': ['hihat', 'hi-hat','hh', 'hat', 'cow', 'tamb', 'wood', 'ride', 'crash', 'crush', 'cymb', 'bell'],
            'clap': ['clap'],
            # 'synth': ['syn'],
            # 'perc': ['perc'],
        }

        # names of the classes
        self.classes = list(self.class_groups.keys())
        
        # mean of the training data (per feature dimension)
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
        
        self.log_faulty_files = []
        self.log_duplicate_labels = []
        self.log_no_label = []
        
        



    def __len__(self):
        return len(self.files)

    def get_label_from_path(self, file):
        fni = file.rfind('/')
        fnd = file.rfind('/', 0, fni)

        filename = file[fni+1:].lower()
        filename_and_parent = file[fnd+1:].lower()
        
        # define class vector
        # check if filename contains class name, if multiple classes are found, choose meta class either randomly or last occurence
        y_vec = torch.zeros(len(self.class_groups.keys()))
        for i, (key, value) in enumerate(self.class_groups.items()):
            in_group = [filename_and_parent.rfind(name)+1.0 if name in filename_and_parent.lower() else 0.0 for name in value]
            y_vec[i] = max(in_group)
        
        # if there are multiple classes present, y_vec includes all of them
        # y_numeric however can contain only one class
        # either select one randomly or select the last occurence of a class in the string
        # this is kinda dirty but for sometimes only one class is needed
        select_randomly = False # randomly or last occurence
        
        y_numeric = torch.tensor([-1]) 
        num_classes_found = len(np.where(y_vec>0.0)[0])     
        
        class_indices = np.where(y_vec>0.0)[0] # which classes are found
        
        if num_classes_found > 0: # at least one class found
            if select_randomly:
                y_numeric = random.choice(class_indices)
            else:
                y_numeric = np.argmax(y_vec) # select the class which is last in string, because this is probably the class, e.g. 'clappy kick' is probably a kick and not a clap
        
        if num_classes_found > 1: # two or more classes found
            print('found multiple classes for name '+filename+' choosing class: '+str(self.classes[y_numeric.item()]))
            # set all classes to 0.0 except the selected one
            # this could be also commented out for training with multiple classes
            y_vec[:] = 0.0
            y_vec[y_numeric] = 1.0
            self.log_duplicate_labels.append([filename_and_parent, class_indices, self.classes[y_numeric.item()]])

        # # using all class occurences for training
        y_vec[y_vec>1.0] = 1.0 # we are saving string positions here, so set all positive classes to 1.0
        
        ## debug check
        if not y_vec.any(): 
            print('couldnt find class for name: '+filename_and_parent)
            self.log_no_label.append(filename_and_parent)


        return y_numeric, y_vec

    def __getitem__(self, idx):
        file = self.files[idx]
        y_numeric, y_vec = self.get_label_from_path(file)
        
        # load audio from file and encode it
        wav_data = self.load_audio(file)
        x, x_transposed = self.encode_sample(wav_data)


        y_vec = y_vec.to(self.device)
        y_numeric = y_numeric.to(self.device)
        return x, x_transposed, y_vec, y_numeric
    
    def load_audio(self, file):
        try:
            data, sample_hz = torchaudio.load(file)
        except:
            print('error loading file '+file)
            data, sample_hz = torch.zeros((1,20000)), 24000 # todo: this is a hack to make it work with the sampler
            self.log_faulty_files.append(file)
            

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # first resample data to the max target freq

        data = resample(data, sample_hz, self.target_sample_hz)

        audio_length = data.size(1)
        
        # unifying the length of samples by cropping too long samples
        # of by filling too short samples with zeros
        if audio_length > self.length:
            data = data[:, 0:self.length]
        else:
            data2 = torch.zeros((1,self.length),dtype=torch.float32)
            data2[0,:data.shape[1]] = data
            data = data2

        # todo: what is this for? Important for saving the audiofile back later?
        data = data.squeeze() # squeeze first dimension with size 1
        
        data = data.to(self.device)
        
        return data

    
    def encode_sample(self,wav_data):
        codes, indices, _ = self.encodec.forward(wav_data, 24000, True)

        # codes = codes / 5.0

        # subtract mean per feature dimension

        # codes = codes - self.fm



        # shift data for target array construction
        # x = 0 1 2 3 4
        # y = 1 2 3 4

        x, y = codes[:-1], codes
        x, y = x.to(device=self.device), y.to(device=self.device)
        x = torch.cat((torch.zeros((1,128)).to(self.device),x),dim=0)
        # y = torch.cat((torch.zeros((1,128)).to(self.device),y),dim=0)
        return x, y

    def decode_sample(self, codes):
        # codes = codes + self.fm
        
        # codes = codes * 5.0

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
