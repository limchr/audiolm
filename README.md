## Conditioning Transformer Models with Spatial Information for Music Sample Generation

### Main Scripts:

- condition_vae_train.py: Train the vae for the 2d embedding
- transformer_train.py: Train the transformer using the pretrained vae as condition
- transformer_generate.py: Generate samples using the transformer in a grid fashion and generate all other data structures for exporting to the web interface

### Important Files:

- audiolm_pytorch/data.py: Dataloading and buffering, encoding and decoding with facebook encodec model
- audiolm_pytorch/condition_vae.py: VAE implementation
- audiolm_pytorch/gpt.py: Transformer implementation based on https://github.com/karpathy/nanoGPT
- audiolm_pytorch/condition_cnn.py: CNN implementation that was used for conditioning before (obsolete)

### Installation:

Project is forked from https://github.com/lucidrains/audiolm-pytorch but we only use their implementation of the encodec model. Their installation instruction should be enough to get the project running, also change absolute paths in experiment_config.py appropriately.