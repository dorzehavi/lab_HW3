import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch_geometric.utils as utils
from dataset import HW3Dataset

# Load dataset
dataset = HW3Dataset(root='data/hw3/')
data = dataset.data
data.y = data.y.squeeze(1)

# 1. Plotting distribution of categories

category_counts = Counter(data.y.cpu().numpy())  # count each category occurrence
categories = list(category_counts.keys())
counts = list(category_counts.values())

plt.figure(figsize=(12,6))
plt.bar(categories, counts)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Categories')
plt.show()

# Convert the edge_index tensor to a NetworkX graph
graph = utils.to_networkx(data, to_undirected=False)

print("Number of edges: ", graph.number_of_edges())

# Calculate degree of each node
in_degree_sequence = [d for n, d in graph.in_degree()]
out_degree_sequence = [d for n, d in graph.out_degree()]
degree_sequence = [d for n, d in graph.degree()]


# Convert the list of degrees to a numpy array for easier calculations
in_degree_sequence = np.array(in_degree_sequence)
out_degree_sequence = np.array(out_degree_sequence)
degree_sequence = np.array(degree_sequence)

# Compute max, min and average degree
in_max_degree = np.max(in_degree_sequence)
out_max_degree = np.max(out_degree_sequence)


# Print the computed values
print("Max IN degree: ", in_max_degree)
print("Max OUT degree: ", out_max_degree)



# Count nodes with degree greater than 100
in_nodes_above_100 = np.sum(in_degree_sequence > 100)
print(f'Number of nodes with IN degree greater than 100: {in_nodes_above_100}')

out_nodes_above_100 = np.sum(out_degree_sequence > 100)
print(f'Number of nodes with OUT degree greater than 100: {out_nodes_above_100}')


# Limit the sequence to 100 for the plot
in_degree_sequence_limited = in_degree_sequence[in_degree_sequence <= 100]
out_degree_sequence_limited = out_degree_sequence[out_degree_sequence <= 100]
degree_sequence_limited = degree_sequence[degree_sequence <= 100]


# Plot histogram
plt.figure(figsize=(12,6))
plt.hist(in_degree_sequence_limited, bins=100)
plt.title('Node IN Degree Distribution (up to degree 100)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()


# Plot histogram
plt.figure(figsize=(12,6))
plt.hist(out_degree_sequence_limited, bins=100)
plt.title('Node OUT Degree Distribution (up to degree 100)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()


# Plot histogram
plt.figure(figsize=(12,6))
plt.hist(degree_sequence_limited, bins=100)
plt.title('Node Degree Distribution (up to degree 100)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()


# Load 'node_year' into a numpy array
node_year = data.node_year.cpu().numpy()

# Plot the distribution of years
plt.figure(figsize=(12, 6))
plt.hist(node_year, bins=50) # You can change the number of bins depending on your data
plt.title('Distribution of Node Years')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Load 'node_year' and 'y' into numpy arrays
node_year = data.node_year.cpu().numpy().squeeze()
labels = data.y.cpu().numpy().squeeze()

# Compute Pearson correlation
correlation = np.corrcoef(node_year, labels)[0, 1]

print("Correlation between node_year and label: ", correlation)
