# Conditioning Transformer Models with Spatial Information for Music Sample Generation

## Main Scripts:

- condition_vae_train.py: Train the vae for the 2d embedding
- transformer_train.py: Train the transformer using the pretrained vae as condition
- transformer_generate.py: Generate samples using the transformer in a grid fashion and generate all other data structures for exporting to the web interface

## Important Files:

- audiolm_pytorch/data.py: Dataloading and buffering, encoding and decoding with facebook encodec model
- audiolm_pytorch/condition_vae.py: VAE implementation
- audiolm_pytorch/gpt.py: Transformer implementation based on https://github.com/karpathy/nanoGPT
- audiolm_pytorch/condition_cnn.py: CNN implementation that was used for conditioning before (obsolete)

## Installation:

Project is forked from https://github.com/lucidrains/audiolm-pytorch but we only use their implementation of the encodec model. Their installation instruction should be enough to get the project running, also change absolute paths in experiment_config.py appropriately.

Installation with conda / pip:

    conda remove --name audiolm --all # if you screw up and have to install again
    conda create --name audiolm
    conda activate audiolm
    conda install python==3.9.2
    pip install -r requirements.txt
    # pip install 'accelerate>=0.24.0' 'beartype>=0.16.1' 'einops>=0.7.0' 'ema-pytorch>=0.2.2' encodec fairseq 'gateloop-transformer>=0.0.24' joblib 'local-attention>=1.9.0' scikit-learn sentencepiece 'torch>=1.12' torchaudio transformers tqdm 'vector-quantize-pytorch>=1.11.3'
    # export LD_LIBRARY_PATH='/home/chris/miniconda3/envs/audiolm/lib/python3.9/site-packages/nvidia/cudnn/lib/;/home/chris/miniconda3/envs/audiolm/lib/python3.9/site-packages/nvidia/nvjitlink/lib/'

## Interesting Links

- https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
