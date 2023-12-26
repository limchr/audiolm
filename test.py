from audiolm_pytorch import SoundStream, SoundStreamTrainer
import torchaudio
from pathlib import Path

ds_path = '/home/chris/data/audio_samples/ds_extracted/'


path = Path(ds_path)
assert path.exists(), f'folder "{str(path)}" does not exist'
exts = ['flac', 'wav', 'mp3', 'webm']
files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]


for file in files:
    print(file)
    try:
        data, sample_hz = torchaudio.load(file)
    except:
        print(file)



soundstream = SoundStream(
    codebook_size = 4096,
    rq_num_quantizers = 8,
    rq_groups = 2,                      # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
    use_lookup_free_quantizer = False,  # whether to use residual lookup free quantization
    attn_window_size = 128,             # local attention receptive field at bottleneck
    attn_depth = 2                      # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = ds_path,
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length_seconds = 2,  # train on 2 second audio
    num_train_steps = 1_000_000
).cuda()

trainer.train()

# after a lot of training, you can test the autoencoding as so

soundstream.eval() # your soundstream must be in eval mode, to avoid having the residual dropout of the residual VQ necessary for training

audio = torch.randn(10080).cuda()
recons = soundstream(audio, return_recons_only = True) # (1, 10080) - 1 channel
