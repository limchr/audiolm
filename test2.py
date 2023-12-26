from audiolm_pytorch import SoundStream, SoundStreamTrainer, soundstream
import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F

from pathlib import Path
from einops import  rearrange, reduce
from audiolm_pytorch.utils import curtail_to_multiple

from audiolm_pytorch import EncodecWrapper




def load_audiofile(file):
    
    target_sample_hz = 16000

    data, sample_hz = torchaudio.load(file)

    assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

    if data.shape[0] > 1:
        # the audio has more than 1 channel, convert to mono
        data = reduce(data, 'c ... -> 1 ...', 'mean')

    # first resample data to the max target freq

    data = resample(data, sample_hz, target_sample_hz)
    sample_hz = target_sample_hz

    # then curtail or pad the audio depending on the max length

    max_length = 32000
    audio_length = data.size(1)

    def exists(val):
        return val is not None
    def cast_tuple(val, length = 1):
        return val if isinstance(val, tuple) else ((val,) * length)

    def is_unique(arr):
        return len(set(arr)) == len(arr)


    if exists(max_length):
        if audio_length > max_length:
            max_start = audio_length - max_length
            start = torch.randint(0, max_start, (1, ))
            data = data[:, start:start + max_length]
        else:
            data = F.pad(data, (0, max_length - audio_length), 'constant')

    data = rearrange(data, '1 ... -> ...')

    # resample if target_sample_hz is not None in the tuple

    num_outputs = 1
    data = cast_tuple(data, num_outputs)

    data_tuple = tuple(resample(d, sample_hz, target_sample_hz) for d, target_sample_hz in zip(data, [target_sample_hz]))

    output = []

    # process each of the data resample at different frequencies individually for curtailing to multiple

    aseq_len_multiple_of = 320
    for data, seq_len_multiple_of in zip(data_tuple, [aseq_len_multiple_of]):
        if exists(seq_len_multiple_of):
            data = curtail_to_multiple(data, seq_len_multiple_of)

        output.append(data.float())

    # cast from list to tuple

    output = tuple(output)

    # return only one audio, if only one target resample freq

    if num_outputs == 1:
        return output[0]

    return output





filea = load_audiofile('/home/chris/data/audio_samples/ds_extracted/[KB6]_Electro-Harmonix_DRM-15/hh1.wav')
fileb = load_audiofile('/home/chris/data/audio_samples/ds_extracted/[KB6]_Electro-Harmonix_DRM-15/hh2.wav')


encodec = EncodecWrapper()

codes, indices, _ = encodec.forward(filea, 16000, True)
codesb, indicesb, _ = encodec.forward(fileb, 16000, True)

encoded = encodec.decode_from_codebook_indices(indices[None])
encodedb = encodec.decode_from_codebook_indices(indicesb[None])
encoded = encodec.decode(codes)

torchaudio.save('file3.wav', fileb.reshape(1,-1), 16000)
torchaudio.save('file4.wav', encoded.reshape(1,-1), 24000)


encoded = encodec.decode([codes])
torchaudio.save('asdf.wav', encoded.reshape(1,-1), 24000)

soundstream = SoundStream.init_and_load_from('./results/soundstream.54000.pt')

nc = encodec

audio = torch.randn(1, 512 * 320)

codes = nc.tokenize(audio)

# you can now train anything with the codebook ids

recon_audio_from_codes = soundstream.decode_from_codebook_indices(codes)


# sanity check

assert torch.allclose(
    recon_audio_from_codes,
    soundstream(audio, return_recons_only = True)
)




codes = soundstream.tokenize(filea)
recon_audio_from_codes = soundstream.decode_from_codebook_indices(codes)
torchaudio.save('file.wav', recon_audio_from_codes.reshape(1,-1), 16000)
torchaudio.save('file2.wav', filea.reshape(1,-1), 16000)
