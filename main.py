import numpy as np
import pandas as pd

sampling_rate = 24414
window_duration = 2e-3


def spike_sorting(data):
    spike_timestamps = []
    mean_spikes = []

    window_size = int(window_duration * sampling_rate)

    for electrode_data in data:
        threshold = 3.5 * np.std(electrode_data[:500])

        spike_indices = np.where(electrode_data > threshold)[0]

        spike_waveforms = []

        for index in spike_indices:
            start_index = max(0, index - window_size // 2)
            end_index = min(len(electrode_data), index + window_size // 2)
            spike_waveform = electrode_data[start_index:end_index]
            spike_waveforms.append(spike_waveform)

        spike_timestamps.append(spike_indices / sampling_rate)

        mean_spike = np.mean(spike_waveforms, axis=0)
        mean_spikes.append(mean_spike)

    return spike_timestamps, mean_spikes


electrode1 = []
electrode2 = []

with open("Data.txt", "r") as file:
    for line in file:
        data = line.strip().split("\t")

        electrode1.append(float(data[0]))
        electrode2.append(float(data[1]))


data = [electrode1, electrode2]

spike_timestamps, mean_spikes = spike_sorting(data)

spike_timestamps_df = pd.DataFrame(spike_timestamps)
mean_spikes_df = pd.DataFrame(mean_spikes)

spike_timestamps_df.transpose().to_csv("spike_timestamps.csv", index=False)
mean_spikes_df.transpose().to_csv("mean_spikes.csv", index=False)
