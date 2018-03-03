import numpy as np
import matplotlib.pyplot as plt

def plot_dist_vs_dims():

    distances = []
    for i in range(1000):
        distance = 0
        for _ in range(100):
            points = np.random.normal(size=(2, i + 1))
            distance += np.sqrt(np.sum((points[0] - points[1]) ** 2))
        distances.append(distance / 100.)

    plt.plot(range(1, 1001), distances)
    plt.xlabel("Dimensionality")
    plt.ylabel("Avg. Distance")