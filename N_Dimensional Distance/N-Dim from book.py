import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

def random_point (dim:int) -> list[float]:
    return (random.random() for _ in range (dim))

def random_distances (dim:int, num_pairs: int) -> list [float]:
    distances = []
    for _ in range(num_pairs):
        # Generate two random points for each pair
        point1 = np.array(list(random_point(dim)))  # Convert generator to a NumPy array
        point2 = np.array(list(random_point(dim)))
        # Compute the Euclidean distance between the two points
        distances.append(np.linalg.norm(point1 - point2))
    return distances

dimensions = range (1,101)

avg_distances = []
min_distances = []
ratio_distances = []


random.seed (0)
for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensions"):
    distances = random_distances(dim,1000)
    avg_distances.append(sum(distances)/1000)
    min_distances.append(min(distances))
    if avg_distances[dim-1] != 0:
        ratio = min_distances[dim-1] / avg_distances [dim-1]
    else:
        ratio = 0
    ratio_distances.append(ratio)

    


plt.figure(figsize=(12, 6))  # Set figure size

# Plot avg_distances and min_distances
plt.plot(dimensions, avg_distances, label='Average Distance', color='blue', marker='o')
plt.plot(dimensions, min_distances, label='Minimum Distance', color='red', marker='x')


# Add labels and title
plt.xlabel('Dimensions')
plt.ylabel('Distance')
plt.title('Curse of Dimensionality: Distance Between Random Points')

# Add a legend
plt.legend()


# Show the plot
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))  # Set figure size for the second plot
plt.plot(dimensions, ratio_distances, label='min_distance / avg_distance', color='green', marker='s')
plt.xlabel('Dimensions')
plt.ylabel('Ratio')
plt.title('Ratio of Minimum to Average Distance')
plt.legend()
plt.grid(True)
plt.show()
