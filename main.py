import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

sampling_rate = 24414
window_duration = 2e-3
window_size = int(window_duration * sampling_rate)
max_peaks_to_plot = 20
samples_to_consider = 20000
num_clusters = 3  # Gotten by the elbow method

def get_features(list):
    data = []
    threshold = 3.5 * np.std(list[:499])

    for index, element in enumerate(list):
        if element > threshold:
            maximum = element
            max_index = index
            for j in range(index, len(list)):
                if list[j] > threshold:
                    if list[j] > maximum:
                        maximum = list[j]
                        max_index = j
                else:
                    break

            start_index = int(max(0, max_index - window_size // 2))
            end_index = int(min(len(list), max_index + window_size // 2))

            window_list = list[start_index:end_index]
            std = np.std(window_list)
            diff = np.max(np.abs(np.diff(window_list)))

            data.append(
                {"standard deviation": std, "difference": diff, "peakIndex": max_index}
            )
    dataDF = pd.DataFrame(data)
    return dataDF

def spike_sorting(data):
    threshold = 3.5 * np.std(data[:499])
    spike_indices, _ = find_peaks(data[:samples_to_consider], height=threshold)

    # Sort spike indices by peak amplitude in descending order
    sorted_spike_indices = sorted(spike_indices, key=lambda index: data[index], reverse=True)

    selected_spike_indices = sorted_spike_indices[:max_peaks_to_plot]

    spikes = []
    for index in selected_spike_indices:
        start_index = int(max(0, index - window_size // 2))
        end_index = int(min(len(data), index + window_size // 2))
        spike_waveform = data[start_index:end_index]
        spikes.append(spike_waveform)

    return selected_spike_indices, spikes

electrode1 = []
electrode2 = []

with open("Data.txt", "r") as file:
    for line in file:
        data = line.strip().split("\t")

        electrode1.append(float(data[0]))
        electrode2.append(float(data[1]))

electrodes_data = [electrode1, electrode2]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, electrode_data in enumerate(electrodes_data):
    spike_indices, _ = spike_sorting(electrode_data)

    # Perform KMeans clustering on the spike indices
    X = np.array(list(zip(spike_indices, [electrode_data[index] for index in spike_indices])))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Plot raw data
    axes[0][i].plot(electrode_data[:samples_to_consider], label=f'Electrode {i + 1} Raw Data')

    # Plot the peaks with different colors for each cluster
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        axes[0][i].plot(X[cluster_indices, 0], X[cluster_indices, 1], '*', markersize=10, label=f'Cluster {cluster_id + 1}')

    axes[0][i].set_xlabel('Sample')
    axes[0][i].set_ylabel('Amplitude')
    axes[0][i].legend()
    axes[0][i].set_title(f'Electrode {i + 1} Raw Data with Clusters')


for i in range(2):
    electrode_DF = get_features(electrodes_data[i])
    data = list(zip(electrode_DF["standard deviation"], electrode_DF["difference"]))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    axes[1][i].scatter(electrode_DF["standard deviation"], electrode_DF["difference"], c=kmeans.labels_)
    axes[1][i].set_xlabel('Standard Deviation')
    axes[1][i].set_ylabel('Difference')
    axes[1][i].set_title(f'Electrode {i + 1} Scatter Plot with Clusters')

plt.tight_layout()
plt.show()
