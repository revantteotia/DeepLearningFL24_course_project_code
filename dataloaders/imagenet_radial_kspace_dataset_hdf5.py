import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
from threading import Thread
from queue import Queue
import time
import json
import pickle

class ImagenetRadialKspaceDatasetHDF5(Dataset):
    def __init__(self, 
                 root_dir, 
                 train=True, 
                 dev=False,
                 log_scale_amplitude=True,
                 log_scale_freq=False, 
                 z_normalize_spokes=False,
                 min_max_normalize=True,
                 use_half_spoke=False,
                 ):
        self.root_dir = root_dir
        self.train = train
        self.log_scale_amplitude = log_scale_amplitude
        self.log_scale_freq = log_scale_freq
        self.z_normalize_spokes = z_normalize_spokes
        self.min_max_normalize = min_max_normalize
        self.data_list = []
        self.use_half_spoke = use_half_spoke

        if self.log_scale_amplitude:
            print("First amplitude is log scaled and then other normalizations are applied")

        if self.min_max_normalize:
            print("min_max_normalize are applied")


        assert not (self.use_half_spoke == True and self.log_scale_freq == True)

        # if doing z norm, other norms are not done. We can do log and then z-norm. but later
        if self.z_normalize_spokes:
            assert not self.log_scale_freq
            assert not self.min_max_normalize

        split = 'train' if self.train else 'val'

        if dev == True:
            split = "dev"

        self.hdf5_filepath = os.path.join(root_dir, f"{split}.h5")
        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_filepath, 'r')
        # data is stored under 'data' and labels under 'labels'
        self.data = self.hdf5_file['data']
        self.labels = self.hdf5_file['labels']

        # Get the number of samples
        self.length = self.data.shape[0]
        self.spoke_length = self.data.shape[-1]
        if self.use_half_spoke:
            self.spoke_length = self.data.shape[-1] // 2
        else:
            self.spoke_length = self.data.shape[-1]
        
        with open(os.path.join(root_dir, "train_data_stats.pkl"), "rb") as fp:
            stats_data = pickle.load(fp)
        self.mag_mean = stats_data["mean"]
        self.mag_std = stats_data["std"]
        self.mean_of_logs = stats_data["mean_of_logs"]
        self.std_of_logs = stats_data["std_of_logs"]

        if self.use_half_spoke:
            self.mag_mean = self.mag_mean[:len(self.mag_mean)//2]
            self.mag_std = self.mag_std[:len(self.mag_std)//2]
            self.mean_of_logs = self.mean_of_logs[:len(self.mean_of_logs)//2]
            self.std_of_logs = self.std_of_logs[:len(self.std_of_logs)//2]


        if self.log_scale_freq:
            # Precompute log-scaled indices for frequency scaling
            self.precomputed_log_indices = self.precompute_log_indices()
    
    def precompute_log_indices(self):
        # Precompute log-scaled indices for positive and negative frequencies
        max_len = self.data.shape[-1]
        pos_indices = np.logspace(0, np.log2(max_len // 2 - 1), num=max_len // 2, base=2, endpoint=False)
        neg_indices = pos_indices[::-1]
        return pos_indices, neg_indices


    def log_scale_freq_radial_spoke(self, spoke):
        neg_freq = spoke[:len(spoke)//2][::-1]  # Reverse low to high freq
        pos_freq = spoke[len(spoke)//2:]

        pos_freq_log = np.interp(self.precomputed_log_indices[0], np.arange(len(pos_freq)), pos_freq)
        # neg_freq_log = np.interp(self.precomputed_log_indices[1], np.arange(len(neg_freq)), neg_freq)[::-1]
        neg_freq_log = np.interp(self.precomputed_log_indices[1], np.arange(len(neg_freq)), neg_freq)

        return np.concatenate((neg_freq_log, pos_freq_log), axis=0)

    # # SUBOPTIMAL but more understandable
    # def log_scale_freq_radial_spoke(self, spoke):
    #     neg_freq = spoke[:len(spoke)//2]
    #     neg_freq = neg_freq[::-1] # making from low freq to high freq
    #     pos_freq = spoke[len(spoke)//2:]
        
    #     pos_freq_log_indices = np.logspace(0, np.log2(len(pos_freq)-1), num=len(pos_freq), base=2, endpoint=False)
    #     neg_freq_log_indices = np.logspace(0, np.log2(len(neg_freq)-1), num=len(neg_freq), base=2, endpoint=False)

    #     pos_interp_func = interp1d(np.arange(len(pos_freq)), pos_freq)
    #     neg_interp_func = interp1d(np.arange(len(neg_freq)), neg_freq)

    #     log_freq_pos_freq = pos_interp_func(pos_freq_log_indices)
    #     log_freq_neg_freq = neg_interp_func(neg_freq_log_indices)
    #     log_freq_neg_freq = log_freq_neg_freq[::-1] # making it again high to low freq

    #     concatenated_log_scaled_spectrum = np.concatenate((log_freq_neg_freq, log_freq_pos_freq), axis=0)
        
    #     return concatenated_log_scaled_spectrum

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        raw_kspace = self.data[idx]
        radial_kspace_data = raw_kspace.copy()

        # print("raw_kspace.shape: ", raw_kspace.shape)
        # print(radial_kspace_data.shape)
        
        # only taking the first half. Because the second half is symmetric
        if self.use_half_spoke:
            raw_kspace = raw_kspace[:, :raw_kspace.shape[-1]//2]
            radial_kspace_data = radial_kspace_data[:, :radial_kspace_data.shape[-1]//2]


        label = self.labels[idx]
        # num_spokes = radial_kspace_data.shape[0]
        # len_spokes = radial_kspace_data.shape[1]

        if self.log_scale_amplitude:
            # print("log scaling amplitude")
            amp_spectrum = np.abs(radial_kspace_data)
            phase_spectrum = np.angle(radial_kspace_data)
            log_amp = np.log(1 + amp_spectrum)
            radial_kspace_data = log_amp * np.exp(1j * phase_spectrum)
        
        if self.min_max_normalize:
            amp_spectrum = np.abs(radial_kspace_data)
            phase_spectrum = np.angle(radial_kspace_data)

            max_amp, min_amp = np.max(amp_spectrum), np.min(amp_spectrum)
            amp_spectrum = (amp_spectrum - min_amp) / (max_amp - min_amp)

            radial_kspace_data = amp_spectrum * np.exp(1j * phase_spectrum)

        if self.log_scale_freq:
            radial_kspace_data = np.apply_along_axis(self.log_scale_freq_radial_spoke, axis=1, arr=radial_kspace_data)

        if self.z_normalize_spokes:
            # print("z normalizing data")
            mag_spectrum = np.abs(radial_kspace_data)
            phase_spectrum = np.angle(radial_kspace_data)

            if self.log_scale_amplitude:
                # print("using mean of logs and std of logs to z normalize")
                normalized_mag = (mag_spectrum - self.mean_of_logs)/(self.std_of_logs*2)
            else:
                # print("using freq wise mean and std to z normalize")
                normalized_mag = (mag_spectrum - self.mag_mean)/(self.mag_std*4 + 1e-8)

            radial_kspace_data = normalized_mag * np.exp(1j * phase_spectrum)
        
        radial_kspace_data = torch.tensor(radial_kspace_data) # make sure this is complex data

        return radial_kspace_data, label, raw_kspace
    
    # def __del__(self):
    #     # important
    #     self.hdf5_file.close()

if __name__ == "__main__":

    # NOTE: this is outdated now
    print("loading data ...")
    dev_dataset = ImagenetRadialKspaceDatasetHDF5(root_dir='/gpfs/data/chopralab/imagenet-large-radial-kspace-hdf5', 
                                              train=True, 
                                            #   dev=True,
                                              log_scale_amplitude=True,
                                              log_scale_freq=True,
                                              normalize_spokes=True,
                                              )
    # dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    print("len(dev_dataset): ", len(dev_dataset))
    radial_kspace_data, label, raw_kspace = dev_dataset[10000]
    print(radial_kspace_data.shape, radial_kspace_data.dtype)
    print("label: ", label, type(label))
    
    print("radial_kspace_data[0][224]: ", radial_kspace_data[0][220:230]) # TODO: analyze the range of values of these vectors

    # Normalized new tensor: no need for log scale now
    plt.imshow(np.absolute(radial_kspace_data.numpy()), cmap='gray')
    # Add color bar to show the scale
    plt.colorbar(label='Value')
    # Save the figure as a PNG file
    plt.savefig('radial_kspace_loaded.png')
