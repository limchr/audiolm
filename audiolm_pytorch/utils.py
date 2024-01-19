import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

import numpy as np

# functions

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













def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult, from_left = False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]

# base class

class AudioConditionerBase(nn.Module):
    pass
