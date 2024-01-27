import torch
import numpy as np
import matplotlib.pyplot as plt

# 2d embedding schaut welches samples ueberlappen bzw. wie viele samples es pro pixel gibt
from audiolm_pytorch.data import get_audio_dataset
from experiment_config import ds_folders, ds_buffer

# get the audio dataset
dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                dump_path= ds_buffer,
                                                                build_dump_from_scratch=False,
                                                                only_labeled_samples=True,
                                                                test_size=0.2,
                                                                equalize_class_distribution=True,
                                                                equalize_train_data_loader_distribution=False, # for testing purposes the true distribution of the dataset is wanted
                                                                batch_size=32,
                                                                seed=1234)
condition_cnn = torch.load('results/cnn_model.pt')
condition_cnn.eval()


out_img = np.zeros((500,500), dtype=np.int64)
for i,d in enumerate(ds_train):
    print('%d/%d' % (i+1, len(ds_train)))
    dx, dy, dz, dznum = d
    dx = dx.view((1,)+dx.shape)
    z = condition_cnn(dx[:,:32,:],True).cpu().detach().numpy()[0]
    zx = ((z[0] + 1.0)/2.0) * out_img.shape[0]
    zy = ((z[1] + 1.0)/2.0) * out_img.shape[1]
    out_img[np.round(zx).astype(np.int64),np.round(zy).astype(np.int64)] += 1
    
pos = plt.imshow(out_img)
plt.colorbar(pos, ax=plt.gca())
plt.savefig('results/condition_distribution.png')
plt.show()