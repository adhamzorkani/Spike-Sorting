import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

class SpikeSorting:
    def __init__(self, sampling_rate, window_size, th_mult, peaks_num):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.th_mult = th_mult
        self.peaks_num = peaks_num

    def spike_sorting(self, data_raw, which_electrode=1):
        # Initial declarations
        electrode_d = data_raw[:, 0]
        threshold = self.th_mult * np.std(electrode_d[:500])
        peaks, _ = find_peaks(electrode_d, height=threshold, distance=int(self.sampling_rate*self.window_size))

        timestamps_a = []
        spikes_means = []
        features_exp = []
        clusters_ident = []
        spikes_organized = []
        spike_features = []
        
        for peak in peaks:
            start, end = int(peak - (self.sampling_rate * self.window_size) / 2), int(peak + (self.sampling_rate * self.window_size) / 2)
            spike = electrode_d[start:end]
            spikes_organized.append(spike)
        
        for spike in spikes_organized:
            std_dev, max_diff = np.std(spike), np.max(np.abs(np.diff(spike)))
            spike_features.append([std_dev, max_diff])
        
        kmeans = KMeans(n_clusters=3, n_init=100, max_iter=100, tol=1e-6, random_state=0)
        clusters = kmeans.fit_predict(spike_features)
        
        timestamps_a.append(peaks / self.sampling_rate)
        spikes_means.append(np.mean(spikes_organized, axis=0))
        features_exp.extend(spike_features)
        clusters_ident.extend(clusters)

        self.plot_spike_data(electrode_d, peaks, clusters, which_electrode)

        return timestamps_a, spikes_means, features_exp, clusters_ident

    def plot_spike_data(self, electrode_d, peaks, clusters, which_electrode=1):
        plt.figure(figsize=(10, 6))
        plt.plot(electrode_d[:20000], label='Raw Data')

        for i, (peak, cluster) in enumerate(zip(peaks[:self.peaks_num], clusters[:self.peaks_num])):
            plt.plot(peak, electrode_d[peak], 'b*' if cluster == 0 else 'g*' if cluster == 1 else 'r*')

        plt.xlabel('Sample ID')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title(f'Electrode {which_electrode} | Raw Data with Spikes Clustered')
        plt.show()

    def plot_spike_means(self, spikes_means, electrode_number):
        for i, mean_spike in enumerate(spikes_means):
            plt.figure(figsize=(10, 4))
            plt.plot(mean_spike, label=f'Neuron {i + 1}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title(f'Average Spike Neuron {i + 1} Electrode {electrode_number}')
            plt.show()

    def plot_spike_features(self, features_exp, clusters_ident, electrode_number):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(np.array(features_exp)[:, 0], np.array(features_exp)[:, 1], c=clusters_ident, cmap='viridis', marker='o', s=20)
        plt.xlabel('Standard Deviation')
        plt.ylabel('Maximum Difference')
        plt.title(f'Spike Features Colored by Clusters - Electrode {electrode_number}')
        plt.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
        plt.show()

spike_sorter = SpikeSorting(sampling_rate=24414, window_size=2e-3, th_mult=3.5, peaks_num=20)
data_raw = np.loadtxt("Data.txt")
timestamps_1, means_1, features_1, clusters_1 = spike_sorter.spike_sorting(data_raw[:, :1])
spike_sorter.plot_spike_features(features_1, clusters_1, 1)
spike_sorter.plot_spike_means(means_1, 1)
spike_sorter_2 = SpikeSorting(sampling_rate=24414, window_size=2e-3, th_mult=3.5, peaks_num=10)
timestamps_2, means_2, features_2, clusters_2 = spike_sorter_2.spike_sorting(data_raw[:, 1:], 2)
spike_sorter_2.plot_spike_features(features_2, clusters_2, 2)
spike_sorter_2.plot_spike_means(means_2, 2)
