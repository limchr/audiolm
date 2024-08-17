"""
@Project: audiogen_demo
@File: visualize_features.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2024/02/03
"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

features_file = 'features_list.pkl'

# Load the features_list from a file
with open(features_file, 'rb') as file:
    features_list = pickle.load(file)

print("Features loaded successfully.")


# # Assuming you want to plot spectral_centroid
# x_coords = [item[0] for item in features_list]
# y_coords = [item[1] for item in features_list]
# spectral_centroids = [item[2]['spectral_centroid'] for item in features_list]
#
# # plt.scatter(x_coords, y_coords, c=spectral_centroids, cmap='viridis')
# plt.figure(figsize=(10, 8))  # Makes the plot larger
# plt.scatter(x_coords, y_coords, c=spectral_centroids, cmap='viridis')  # Adjust 's' for smaller point size
# plt.colorbar(label='Spectral Centroid')
# plt.xlabel('Index X')
# plt.ylabel('Index Y')
# plt.title('Spectral Centroid Distribution')
# plt.show()

def plot_feature_heatmap(features_list, feature_name, xlabel='', ylabel='', title=None):
    """
    Plots a heatmap for a specific feature across the samples grid.

    Parameters:
    - features_list: List of tuples containing (x, y, features_dict), where features_dict contains feature values.
    - feature_name: The name of the feature to plot (e.g., 'spectral_centroid').
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    - title: Plot title. If None, a default title based on the feature_name will be used.
    """
    # Determine grid size
    max_x = max(features_list, key=lambda item: item[0])[0]
    max_y = max(features_list, key=lambda item: item[1])[1]
    grid_size_x, grid_size_y = max_x + 1, max_y + 1
    feature_grid = np.full((grid_size_x, grid_size_y), np.nan)  # Use NaN for missing data

    # Populate the grid with the specified feature
    for x, y, features in features_list:
        if feature_name in features:
            feature_grid[x, y] = np.log(0.1+features[feature_name])

    # normalizing log between 0 and 1
    feature_grid = (-feature_grid.min()+feature_grid)/(feature_grid.max()-feature_grid.min())

    from PIL import Image
    
    cm = plt.get_cmap('rainbow')
    colored_image = cm(feature_grid)

    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save('results/feature_im_%s.png' % feature_name)

    # Create edges for pcolormesh
    x_edges = np.arange(grid_size_x + 1) - 0.5
    y_edges = np.arange(grid_size_y + 1) - 0.5

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_edges, y_edges, feature_grid.T, cmap='viridis', shading='auto')  # Ensure shading is set for smoother color transition
    plt.gca().invert_yaxis()
    plt.colorbar(label=feature_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f'{feature_name} heatmap')
    plt.tight_layout()
    # remove ticks
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.savefig('results/features_heatmap_%s.png' % feature_name)
    plt.show()

feature_names = ['energy', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate', 'tsne']

for feature_name in feature_names:
    plot_feature_heatmap(features_list, feature_name)
