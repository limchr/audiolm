#/bin/sh

rsync -av /home/chris/data/audio_samples/ chris@gpu7:/home/chris/data/audio_samples
rsync -av /home/chris/miniconda3/ chris@gpu7:/home/chris/miniconda3
rsync -av /home/chris/src/audiolm/ chris@gpu7:/home/chris/src/audiolm