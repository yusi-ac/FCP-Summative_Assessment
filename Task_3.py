import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys

class Queue:

	def __init__(self):
		self.queue = []

	def push(self,item):
		self.queue.append(item)

	def pop(self):
		return self.queue.pop(0)

	def is_empty(self):
		return len(self.queue) == 0

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes = None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes


	def get_mean_degree(self):
		total_degree = 0

		for node in self.nodes:
			for i in node.connections:
				if i == 1:
					total_degree += 1
				else:
					continue
		total_nodes = len(self.nodes)
		mean_degree = total_degree / total_nodes

		return mean_degree

	def add_node(self, node):
		self.nodes.append(node)

	def get_mean_clustering(self):
		total_node_clustering = 0

		for node_index in range(len(self.nodes)):

			node = self.nodes[node_index]
			actual_connections = 0

			neighbours = [self.nodes[neighbour] for neighbour, connection in enumerate(node.connections) if connection]

			possible_connections = len(neighbours) * (len(neighbours) - 1) / 2

			for i, j in enumerate(neighbours):
				for next_to in neighbours[i + 1:]:
					if j.connections[next_to.index]:
						actual_connections += 1

			if possible_connections != 0:
				node_clustering = actual_connections / possible_connections
				total_node_clustering += node_clustering

			mean_clustering = total_node_clustering / len(self.nodes)

		return mean_clustering


	def get_mean_path_length(self):

		total_path_length = 0
		total_paths = 0

		for node in self.nodes:
			path = search_paths(self, node)
			for path_length in path.values():

				total_path_length += path_length
				total_paths += 1

		mean_path_length = total_path_length / total_paths

		return round(mean_path_length, 15)

	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def search_paths(network, start_node):

	#to search paths to all other nodes, return the distance to each other nodes

	paths = {}
	queue = Queue()
	queue.push((start_node,0))
	visited = []
	visited.append(start_node.index)

	while not queue.is_empty():
		start_node, distance = queue.pop()

		for neighbour_index, connection in enumerate(start_node.connections):
			if connection and neighbour_index not in visited:
				neighbour = network.nodes[neighbour_index]
				paths[neighbour.index] = distance + 1
				queue.push((neighbour, distance + 1))
				visited.append(neighbour_index)

	return paths

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")

def parse(arg):
	network = 0
	test_network = 0

	if 'network' == arg[0]:
		network = 1

	if '-test_network' == arg[0]:
		test_network = 1

	return network, test_network

def flags():
	flag = argparse.ArgumentParser(description="type your flags")

	flag.add_argument('-network', action='store_true')
	flag.add_argument('-test_network', action='store_true')


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main(args):
	'''
	this program contains 2 flags, -network and -test_network,
	if -network is used, a variable is also to be input in terminal,
	following the -network flag, representing the specified network size
	'''
	network, test_network = parse(args)

	if network and test_network == 0 and args[1]:
		network_size = args[2]
		connectivity_p = 0.5
		Network.make_random_network(network_size, connectivity_p)
		print('Mean degree:', Network.get_mean_degree())
		print('Average path length:', Network.get_mean_path_length())
		print('Clustering coefficient:', Network.get_mean_clustering())
		Network.plot()

	elif test_network and network == 0:
		test_networks()

	else:
		print('flag input incorrect')

if __name__=="__main__":
	main(sys.argv[1:])