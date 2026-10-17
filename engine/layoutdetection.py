import numpy as np

# 1-NN algorithm that predicts the category of design given the target canvas size.

data = {
    "Social Media": (1080, 1080),
    "Poster": (2400, 3600),
    "Magazine": (800, 1100),
    "Leaderboard": (728, 90),
    "Skyscraper": (160, 600),
}

def feature(width, height):
    # feature 1: log(A) - where A = Aspect ratio (w/h)
    f1 = np.log(width) - np.log(height)
    # feature 2: log(S) - where S = Shape (sqrt(w*h))
    f2 = 0.5*(np.log(width) + np.log(height))

    return np.array([f1, f2])

def classify(width, height, alpha=0.8, beta=0.2):
    # novel point
    x = feature(width, height)

    distances = {}
    for label, (w, h) in data.items():
        rep = feature(w, h)

        distance = np.sqrt((alpha**2)*((x[0] - rep[0])**2)+(beta**2)*((x[1] - rep[1])**2))
        distances[label] = distance

    argmin = min(distances, key=distances.get)
    min_distance = distances[argmin]

    return argmin, min_distance

# test
print(classify(1000, 1200))
print(classify(20000, 5000))
print(classify(1480, 1420))

# input cases
print(classify(1080, 1350))
print(classify(1600, 400))
print(classify(728, 90))