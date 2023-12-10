import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sampling_rate = 24414
window_duration = 2e-3
window_size = int(window_duration * sampling_rate)


def spike_sorting(list):
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


electrode1 = []
electrode2 = []

with open("Data.txt", "r") as file:
    for line in file:
        data = line.strip().split("\t")

        electrode1.append(float(data[0]))
        electrode2.append(float(data[1]))

electrode1_DF = spike_sorting(electrode1)
electrode2_DF = spike_sorting(electrode2)

data1 = list(zip(electrode1_DF["standard deviation"], electrode1_DF["difference"]))

# inertias1 = []
# for i in range(1, 101):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data1)
#     inertias1.append(kmeans.inertia_)

# plt.plot(range(1, 101), inertias1, marker="o")
# plt.title("Elbow method")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")

kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(data1)

fig, ax = plt.subplots()
plt.scatter(
    electrode1_DF["standard deviation"], electrode1_DF["difference"], c=kmeans1.labels_
)


data2 = list(zip(electrode2_DF["standard deviation"], electrode2_DF["difference"]))

# inertias2 = []
# for i in range(1, 101):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data2)
#     inertias2.append(kmeans.inertia_)

# plt.plot(range(1, 101), inertias2, marker="o")
# plt.title("Elbow method")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(data2)

fig, ax = plt.subplots()
plt.scatter(
    electrode2_DF["standard deviation"], electrode2_DF["difference"], c=kmeans2.labels_
)

# fig, ax = plt.subplots()
# plt.plot(electrode1_DF["standard deviation"], electrode1_DF["difference"], "*")
# ax.set_title("Electrode 1 Neuron clusters")
# ax.set_xlabel("standard deviation")
# ax.set_ylabel("Mean")

# fig, ax = plt.subplots()
# plt.plot(electrode2_DF["standard deviation"], electrode2_DF["difference"], "*")
# ax.set_title("Electrode 2 Neuron clusters")
# ax.set_xlabel("standard deviation")
# ax.set_ylabel("Mean")

plt.show()
