# Import the libraries
#working code as of 8/13/2023
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the SOM
grid_size = 100 # size of the grid
n_features = 4 # number of features
learning_rate = 0.5 # initial learning rate
sigma = 1.0 # initial neighborhood size
decay_rate = 0.99 # decay rate for learning rate and neighborhood size
max_iter = 100 # maximum number of iterations

# Load the iris data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = np.genfromtxt(url, delimiter=',', usecols=(0,1,2,3))
target = np.genfromtxt(url, delimiter=',', usecols=(4), dtype=str)

# Normalize the data
data = (data - data.min()) / (data.max() - data.min())

# Initialize the weight vectors randomly
weights = np.random.rand(grid_size, grid_size, n_features)

# Define the neighborhood function as a Gaussian function
def neighborhood(i, j, bmu, sigma):
    return np.exp(-((i-bmu[0])**2 + (j-bmu[1])**2) / (2 * (sigma**2 + 1e-10)))


# Define the decay function for learning rate and neighborhood size
def decay(value, iteration):
    return value * (decay_rate ** iteration)

# Define a function to calculate the MSE and accuracy given the data, target and weights
def evaluate(data, target, weights):
    # Initialize the MSE and accuracy variables
    mse = 0
    acc = 0
    # Loop over each data point
    for x, y in zip(data, target):
        # Find the BMU for the data point
        distances = np.sqrt(np.sum((weights - x)**2, axis=2))
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        # Add the squared distance to the MSE variable
        mse += distances[bmu]**2
        # Find the most frequent label in the BMU's neighborhood
        labels = []
        for i in range(max(0,bmu[0]-1), min(grid_size,bmu[0]+2)):
            for j in range(max(0,bmu[1]-1), min(grid_size,bmu[1]+2)):
                labels.append(target[np.argmin(np.sum((weights[i,j] - data)**2, axis=1))])
        label = max(set(labels), key=labels.count)
        # Compare the label with the true label and update the accuracy variable
        if label == y:
            acc += 1
    # Return the MSE and accuracy as a tuple
    return mse / len(data), acc / len(data)

# Initialize an empty list to store the MSE and accuracy values for each iteration
metrics = []

# Loop over the data for a fixed number of iterations
for t in range(max_iter):
    # Shuffle the data
    np.random.shuffle(data)
    # Loop over each data point
    for x in data:
        # Find the best matching unit (BMU) for the data point
        distances = np.sqrt(np.sum((weights - x)**2, axis=2))
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        # Update the weight vectors of the BMU and its neighbors
        for i in range(grid_size):
            for j in range(grid_size):
                weights[i,j] += learning_rate * neighborhood(i,j,bmu,sigma) * (x - weights[i,j])
        # Decrease the learning rate and neighborhood size over time
        learning_rate = decay(learning_rate, t)
        sigma = decay(sigma, t)
    # At the end of each iteration, call the evaluate function and append the results to the metrics list
    metrics.append(evaluate(data, target, weights))

# Calculate the final accuracy and MSE
final_mse, final_accuracy = evaluate(data, target, weights)

# Print the final accuracy and MSE
print("Final Mean Squared Error:", final_mse)
print("Final Accuracy:", final_accuracy)

# After the main loop, plot the metrics list using matplotlib
metrics = np.array(metrics)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(metrics[:,0])
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.subplot(1,2,2)
plt.plot(metrics[:,1])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()

# Visualize the SOM map with different colors for each class
plt.figure(figsize=(10, 10))
colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
markers = {'Iris-setosa': 'o', 'Iris-versicolor': 's', 'Iris-virginica': 'D'}
for i, x in enumerate(data):
    w = np.unravel_index(np.argmin(np.sqrt(np.sum((weights - x)**2, axis=2))), distances.shape)
    plt.plot(w[0]+0.5, w[1]+0.5, markers[target[i]], markerfacecolor='None',
             markeredgecolor=colors[target[i]], markersize=10, markeredgewidth=2)
plt.axis([0, weights.shape[0], 0, weights.shape[1]])
plt.show()

# Visualize the SOM map in a 3D space
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
for i, x in enumerate(data):
    w = np.unravel_index(np.argmin(np.sqrt(np.sum((weights - x)**2, axis=2))), distances.shape)
    ax.scatter(w[0]+0.5, w[1]+0.5, metrics[i, 0], c=colors[target[i]], marker='o', s=50, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('MSE')
plt.show()
