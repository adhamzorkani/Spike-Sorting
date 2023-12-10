# Spike Sorting Project Report: Digital Signal Processing Course

## Introduction

This project, conducted as a part of CSCE 363/3611 at The American University in Cairo, Fall 2023, aims to apply spike sorting techniques to neural data obtained from two electrodes. Spike sorting is a critical process in neural signal analysis, enabling the differentiation and categorization of neuronal spikes. This report details the methodology and findings from the application of spike sorting using K-means clustering.

## Team Members
- Ahmed Eltokhy
- Adham Elzorkani

## Objectives

- To detect and align spikes in neural data using a threshold-based approach.
- To extract relevant features from the spikes for further analysis.
- To classify spikes using K-means clustering.

## Methodology

### Data Preprocessing

The raw neural data from "Data.txt" was preprocessed by loading it into a 2D array called data_raw. This step was essential to handle the data from each electrode independently.

### Feature Extraction

Feature extraction was carried out on the identified spikes. The criteria for spike detection involved a threshold set based on the standard deviation of the data. For each spike, features such as the standard deviation, the difference between successive samples, and the peak index were extracted.

### Clustering

The K-means clustering algorithm was employed to classify the spikes. The number of clusters was predetermined using the elbow method, and clustering was executed using the `KMeans` class from the scikit-learn library. The K value was decided based on the elbow method, where we choose the number of clusters based on where the elbow of the graph is located.

## How To Run

Simply, clone the repository or download the attached code and the required libraries (pandas, matplotlib, scipy, sklearn, numpy) and run
```
python main.py
```

You can find the repository here:
https://github.com/adhamzorkani/Spike-Sorting/

## Results

**![](https://lh7-us.googleusercontent.com/qrcu8Z4GpS1NoAoH1VUg-uWgqQFNik8Dd5RVJydgBLhiziQkYWMQzTyrhdQVqo_AHvnijpoPW42ZjXOyP7vCLhBAWP5Gn-BQJ8mtQSBQt3XWNfxXcuT7SdsICZsJmsA9YnoLIA5jKmygak7vhs7HaFw)**
**![](https://lh7-us.googleusercontent.com/G0V5YnOJN9hHKE8B_mGcxhB9M0LGGnKOgPDLOvFot7SJLozcT9tdIkDBcNNB44x-iVgtjZnrQlrlxErLeTVISx1YoxZmZ4lq08GkLFVAjIbshh4qfVRViKSE7Xfeqy1UfObzS_TLxDpkrJTCGUdiTPU)**
**![](https://lh7-us.googleusercontent.com/yzn4MMBXaGcOWbmyLr4R0grstu-U2XAMSbrCsJXQcT4zWk8Sf1QI4Y3KEyMCOVMx2skRIq7jJbkxj-iq0bTe6e2Xp_tXripo5kg9d_YMwSsp8Jv3qplamYwcWd3TmHo9-TubioXLcq3MT_TrzCeAA10)**
**![](https://lh7-us.googleusercontent.com/xOlk8rHMiQlFGOmDzDBIeK6Veo0ycgNf7iVmCDurvGWWNl6s49KQy4XG8re0FKwSlQYhTI4RmSOHbXNek6vpbUVVvreb7igL6GE96Py_YGRhqdBH-bezUBdmkzfFD2J7wn7xgvSSXKf85KzYdw-bHh4)**
**![](https://lh7-us.googleusercontent.com/osKI3u_l_d3hykK_rihjZWnTh8pQvU8jAzB_aT-hB-P9nwNq5QelQJ6h8ptHm44ZnadmhPOPyrccsQHphYZXL6BM9lIajbNXhj8PGOHOSXqsWRt6p3hYqM1vQzYocNb5D6bZcKIjk3ecYXFlwPxHiEo)**
**![](https://lh7-us.googleusercontent.com/CXIzknO3K0sBvbKEP9NA-URGrJJ1zKTFaT5vGSBfLQDGz9J9rSb9PaeiKfpoSN23SBs1wyJOE4eE2twvAFNHh1c9Sk407g8AZFtS8Q80Yk8xxHLgp5TGWxJOMN9m3ph7HEc8ALxHk3Ar086AwmExrGs)**


We have determined the number of clusters to be three visually but yet used the elbow method graphing to verify the number of clusters. Then, we used matplotlib to graph both the scatter plot showing the clusters and spikes in the raw electrode data graphs. Some parameters in the code were left globally for tweaking (including sampling rate, window duration, etc.). However, we have used the provided information in the project requirements.

We have also wrote it using OOP to achieve proper modularity and facilitate future edits.

## Conclusion

The project demonstrates the effective application of spike sorting techniques in neural data analysis. The findings underscore the potential of digital signal processing in biomedical signal processing. Future work could explore more advanced algorithms for feature extraction and clustering to enhance the accuracy and efficiency of spike sorting.
