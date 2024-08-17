"""
@Project: audiogen_demo
@File: timbre_features.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2024/02/03
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle


def compute_timbre_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Compute features
    energy = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # Return computed features
    return {
        'energy': np.mean(energy),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        # 'spectral_contrast': np.mean(spectral_contrast, axis=1),
        'spectral_flatness': np.mean(spectral_flatness),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zero_crossing_rate': np.mean(zero_crossing_rate)
    }

def extract_indices(filename):
    # Extract x and y from filename format "generated_{x}_{y}.wav"
    parts = filename.split('_')
    x = int(parts[1])
    y = int(parts[2].split('.')[0])
    return x, y


folder_path = '/home/chris/src/audiogen_demo/data/models/drums/samples'
features_list = []

# Loop through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".wav"):
        file_path = os.path.join(folder_path, file)

        # Compute timbre features
        features = compute_timbre_features(file_path)

        # Extract indices for plotting
        x, y = extract_indices(file)

        # Append features with indices for plotting
        features_list.append((x, y, features))

# At this point, `features_list` contains all the computed features along with indices

# Assuming `features_list` is your list of features
features_file = 'features_list.pkl'

features_vec = [list(x[2].values()) for x in features_list]
# features_vec = [list(x[:2]) + list(x[2].values()) for x in features_list]
features_vec = np.array(features_vec)

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

X_embedded = TSNE(n_components=1, learning_rate='auto', init='random', perplexity=3).fit_transform(features_vec)

for i in range(len(features_list)):
    features_list[i][2]['tsne'] = X_embedded[i]


feature_names = list(features_list[0][2].keys())

x_coords = [item[0] for item in features_list]
y_coords = [item[1] for item in features_list]
for fn in feature_names:
    feats = [item[2][fn] for item in features_list]

    # For testing the extracted features
    # plt.scatter(x_coords, y_coords, c=spectral_centroids, cmap='viridis')
    plt.figure(figsize=(10, 8))  # Makes the plot larger
    plt.scatter(x_coords, y_coords, c=feats, cmap='viridis')  # Adjust 's' for smaller point size
    plt.colorbar(label=fn)
    plt.xlabel('Index X')
    plt.ylabel('Index Y')
    plt.title(fn)
    plt.savefig('results/feature_vis__%s.png' % (fn,))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 8))  # Makes the plot larger
    plt.hist(feats, bins=30, edgecolor='black')
    plt.xlabel(fn)
    plt.ylabel('Frequency')
    plt.title('Real-Valued Feature Distribution of '+fn)
    plt.savefig('results/feature_distr__%s.png' % (fn,))
    plt.show()
    plt.close()
    
    clip_percentile = 5
    lower_bound = np.percentile(feats, clip_percentile)
    upper_bound = np.percentile(feats, 100 - clip_percentile)

    clipped_vector = np.clip(feats, lower_bound, upper_bound)

    scaler = MinMaxScaler()
    normalized_vector = scaler.fit_transform(clipped_vector.reshape(-1, 1)).flatten()
    
    # update
    
    for i in range(len(features_list)):
        features_list[i][2][fn] = normalized_vector[i]


    # plot again
    plt.figure(figsize=(10, 8))  # Makes the plot larger
    plt.scatter(x_coords, y_coords, c=normalized_vector, cmap='viridis')  # Adjust 's' for smaller point size
    plt.colorbar(label=fn)
    plt.xlabel('Index X')
    plt.ylabel('Index Y')
    plt.title(fn)
    plt.savefig('results/feature_normvis__%s.png' % (fn,))
    plt.show()
    plt.close()



    plt.figure(figsize=(10, 8))  # Makes the plot larger
    plt.hist(normalized_vector, bins=30, edgecolor='black')
    plt.xlabel(fn)
    plt.ylabel('Frequency')
    plt.title('Clipped and normalized Real-Valued Feature Distribution of '+fn)
    plt.savefig('results/feature_normdistr__%s.png' % (fn,))
    plt.show()
    plt.close()
    



# Save the features_list to a file
with open(features_file, 'wb') as file:
    pickle.dump(features_list, file)

print(f"Features saved to {features_file}")
