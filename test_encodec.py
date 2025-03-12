import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
from audiolm_pytorch import EncodecWrapper
from einops import rearrange, reduce


# initialize encodec model
encodec = EncodecWrapper()
encodec.to(device='cuda')


# read audio
data, sample_hz = torchaudio.load("ts9_test1_in_FP32.wav")


# process audio
assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'
if data.shape[0] > 1:
    data = reduce(data, 'c ... -> 1 ...', 'mean')
data = resample(data, sample_hz, 24000)
data = rearrange(data, '1 ... -> ...')


# encode
codes, indices, _ = encodec.forward(data.to('cuda'), 24000, True)


# decode
wav = encodec.decode(codes[None])


# save back to file
wav_float = wav.cpu().detach().float()
torchaudio.save('en_decoded_ts9_test1_in_FP32.wav', wav_float.reshape(1, -1), 24000, encoding="PCM_F", bits_per_sample=32)