import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

def distance(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def mean(cluster):
    sum_x = 0
    sum_y = 0
    for p in cluster:
        sum_x += p[0]
        sum_y += p[1]
    return [sum_x/len(cluster), sum_y/len(cluster)]

def initialize_centroids(K, low, high):
    centroid = []
    for i in range(K):
        x = random.uniform(low, high)
        y = random.uniform(low, high)
        centroid.append([x, y])
    return centroid

def assign_clusters(K, centroid, data):
    clusters = [[] for i in range (K)]
    for p in data:
        min = 100000
        for i in range(K):
            dist = distance(p[0] , centroid[i][0], p[1] , centroid[i][1])
            if dist < min:
                min = dist 
                cluster = i
        clusters[cluster].append(p)
    return clusters

def main():
    K=3
    epoch = 5

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_data.csv")
    data = pd.read_csv(csv_path).values

    low = data.min()
    high = data.max()
    centroid = initialize_centroids(K, low, high)

    for i in range(epoch):
        clusters = assign_clusters(K, centroid, data)
        old_centroid = [c[:] for c in centroid]

        for k in range(K):
            if len(clusters[k]) > 0:
                centroid[k]=mean(clusters[k])

        print(f"Epoch {i+1} Centroids:", centroid)

        total_shift = sum(
            distance(old_centroid[k][0], centroid[k][0], old_centroid[k][1], centroid[k][1])
            for k in range(K)
        )
        if total_shift < 1e-6:
            print(f"Converged after epoch {i+1}, stopping early.")
            break

    plot_clusters(clusters, centroid)

def plot_clusters(clusters, centroid):
    for k, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, label=f"Cluster {k+1}")

    centroid_x = [c[0] for c in centroid]
    centroid_y = [c[1] for c in centroid]
    plt.scatter(centroid_x, centroid_y, c="black", marker="X", s=150, label="Centroids")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()

main()