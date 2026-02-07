import time

import numpy as np
import pandas as pd
import torch

from .phySenseTexture.phySenseTexture import phySenseTexture


class BayesianUnary:
    def __init__(self, file_path: str, num_size_x_bin, num_size_y_bin, num_size_z_bin,
                 model_path, label_encoder_path,
                 edge_buffer_ratio, num_regions, region_size_ratio, workers_fraction=1):
        self.dataset = pd.read_json(file_path)
        self.num_size_x_bin = num_size_x_bin
        self.num_size_y_bin = num_size_y_bin
        self.num_size_z_bin = num_size_z_bin
        self.size_x_bin = None
        self.size_y_bin = None
        self.size_z_bin = None
        self.size_x_range = None
        self.size_y_range = None
        self.size_z_range = None
        self.texture_predictor = phySenseTexture(model_path, label_encoder_path,
                                                 num_regions, region_size_ratio, edge_buffer_ratio, workers_fraction)

    def calculate_relative_frequencies(self, feature, num_bins, min_bin_norm, max_bin_norm):
        bins = np.linspace(min_bin_norm, max_bin_norm, num_bins + 1, dtype=np.float32)  # Creating bins
        self.dataset[feature + '_bin'] = pd.cut(self.dataset[feature + '_norm'], bins=bins, include_lowest=True)

        # Calculating the relative frequency of each tracking name within each bin
        relative_frequencies = self.dataset.groupby([feature + '_bin', 'tracking_name']).size().unstack(fill_value=0)
        sums = relative_frequencies.sum(axis=1)
        relative_frequencies_normalized = relative_frequencies.div(sums.where(sums != 0, 1), axis=0)

        return relative_frequencies_normalized

    def create_bins(self):
        self.dataset[['size_x_norm', 'size_y_norm', 'size_z_norm']] = pd.DataFrame(self.dataset['size'].tolist(),
                                                                                   index=self.dataset.index)
        self.size_x_range = [self.dataset['size_x_norm'].min(), self.dataset['size_x_norm'].max()]
        self.size_y_range = [self.dataset['size_y_norm'].min(), self.dataset['size_y_norm'].max()]
        self.size_z_range = [self.dataset['size_z_norm'].min(), self.dataset['size_z_norm'].max()]

        self.size_x_bin = self.calculate_relative_frequencies('size_x',
                                                              self.num_size_x_bin,
                                                              self.size_x_range[0],
                                                              self.size_x_range[1])
        self.size_y_bin = self.calculate_relative_frequencies('size_y',
                                                              self.num_size_y_bin,
                                                              self.size_y_range[0],
                                                              self.size_y_range[1])
        self.size_z_bin = self.calculate_relative_frequencies('size_z',
                                                              self.num_size_z_bin,
                                                              self.size_z_range[0],
                                                              self.size_z_range[1])
        self.linespace_x_bins = np.linspace(self.size_x_range[0], self.size_x_range[1], self.num_size_x_bin + 1,
                                            dtype=np.float32)
        self.linespace_y_bins = np.linspace(self.size_y_range[0], self.size_y_range[1], self.num_size_y_bin + 1,
                                            dtype=np.float32)
        self.linespace_z_bins = np.linspace(self.size_z_range[0], self.size_z_range[1], self.num_size_z_bin + 1,
                                            dtype=np.float32)

    # Function to predict probabiliti1es in the corresponding bin
    def predict_prob(self, my_bin, relative_frequencies_train):
        if my_bin in relative_frequencies_train.index:
            probabilities = relative_frequencies_train.loc[my_bin]
            return probabilities  # Returning the probabilities
        else:
            return None  # In case there is no matching bin

    def inference(self,
                  size_x: torch.Tensor,
                  size_y: torch.Tensor,
                  size_z: torch.Tensor,
                  images_list):
        """
        :param size_x: [batch_size]
        :param size_y: [batch_size]
        :param size_z: [batch_size]
        :param filename_list: [batch_size]
        :return: [4, batch_size, num_class]
        """
        size_x_pd = pd.Series(size_x.cpu().numpy())
        size_x_pd = size_x_pd.clip(lower=self.size_x_range[0] + 1e-4, upper=self.size_x_range[1] - 1e-4)
        size_x_bin_x = pd.cut(size_x_pd, bins=self.linespace_x_bins, include_lowest=True)

        size_y_pd = pd.Series(size_y.cpu().numpy())
        size_y_pd = size_y_pd.clip(lower=self.size_y_range[0] + 1e-4, upper=self.size_y_range[1] - 1e-4)
        size_y_bin_x = pd.cut(size_y_pd, bins=self.linespace_y_bins, include_lowest=True)

        size_z_pd = pd.Series(size_z.cpu().numpy())
        size_z_pd = size_z_pd.clip(lower=self.size_z_range[0] + 1e-4, upper=self.size_z_range[1] - 1e-4)
        size_z_bin_x = pd.cut(size_z_pd, bins=self.linespace_z_bins, include_lowest=True)

        # Applying the prediction function to the test set
        predicted_size_x = pd.DataFrame([self.predict_prob(my_bin, self.size_x_bin) for my_bin in size_x_bin_x])
        predicted_size_x.reset_index(drop=True, inplace=True)

        predicted_size_y = pd.DataFrame([self.predict_prob(my_bin, self.size_y_bin) for my_bin in size_y_bin_x])
        predicted_size_y.reset_index(drop=True, inplace=True)

        predicted_size_z = pd.DataFrame([self.predict_prob(my_bin, self.size_z_bin) for my_bin in size_z_bin_x])
        predicted_size_z.reset_index(drop=True, inplace=True)
        
        predicted_probs_texture = self.texture_predictor.predict_prob(images_list)
            
        if predicted_size_x.shape[1] == 3:
            LabelSpace = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']
            for i, label in enumerate(LabelSpace):
                if label not in predicted_size_x.columns.to_list():
                    predicted_size_x.insert(i, label, 0)
                    predicted_size_y.insert(i, label, 0)
                    predicted_size_z.insert(i, label, 0)
        if predicted_probs_texture.shape[1] == 3:
            LabelSpace = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']
            for i, label in enumerate(LabelSpace):
                if label not in predicted_probs_texture.columns.to_list():
                    predicted_probs_texture.insert(i, label, 0)

        return [predicted_size_x, predicted_size_y, predicted_size_z, predicted_probs_texture]
