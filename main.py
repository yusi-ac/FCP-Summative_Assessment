import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import argparse
import sys

class Queue: #for breadth-first-search

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

	def update_value(self, neighbor_value, threshold, beta):
		#####task 2#####
		if abs(self.value - neighbor_value) <= threshold:
			self.value += beta * (neighbor_value - self.value)

class Network: 
	#####For task 3 & 4, as the functions are defined under class#####
	
	def __init__(self, nodes=None):
		#given code
		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def update_nodes(self, nds):
		#####task 2#####
		if len(self.nodes) == len(nds):
			self.nodes = nds
		else:
			print('Node length not equal')

	def get_mean_degree(self):
		#####task 3#####
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

	def get_mean_clustering(self):
		#####task 3#####
		total_node_clustering = 0

		for node_index in range(len(self.nodes)): #for each node

			node = self.nodes[node_index]
			actual_connections = 0

			neighbours = [self.nodes[neighbour] for neighbour, connection in enumerate(node.connections) if connection]

			possible_connections = len(neighbours) * (len(neighbours) - 1) / 2
			#calculate possible connections

			for i, j in enumerate(neighbours): #and check each of its neighbours to see the actual connections
				for next_to in neighbours[i + 1:]:
					if j.connections[next_to.index]:
						actual_connections += 1

			if possible_connections != 0: #calculate clustering for current node and add value to totals
				node_clustering = actual_connections / possible_connections
				total_node_clustering += node_clustering

			mean_clustering = total_node_clustering / len(self.nodes)

		return mean_clustering
	
	def get_mean_path_length(self):
		#####task 3#####
		total_path_length = 0
		total_paths = 0

		for node in self.nodes:
			path = search_paths(self, node) #function defined as this algorithem is used many times for each node
			for path_length in path.values():

				total_path_length += path_length
				total_paths += 1 #the problem of counting path to current node itself is solved in search_paths()

		mean_path_length = total_path_length / total_paths

		return round(mean_path_length, 15) #due to test requirements of 15dp
	 
	def make_random_network(self, N, connection_probability=0.5): 
		#given code
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

	

	def make_ring_network(self, N, neighbour_range=1):
		#####task 4#####
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for i in range(-neighbour_range, neighbour_range):
				if i == 0:
					continue

				neighbour_index = (index + i + N) % N
				node.connections[neighbour_index] = 1
				self.nodes[neighbour_index].connections[index] = 1
			
	
	def make_small_world_network(self, N, re_wire_prob=0.2):
		#####task 4#####
		neighbour_range = 2
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for i in range(-neighbour_range, neighbour_range):
				if i == 0:
					continue
		
				neighbour_index = (index + i + N) % N

				if np.random.random() < re_wire_prob:
					random_neighbour_index = 0
					random_times = 100
					while random_times > 0:
						random_neighbour_index = np.random.randint(0, N)
						if random_neighbour_index != index and node.connections[random_neighbour_index] != 1:
							neighbour_index = random_neighbour_index
							break
						random_times = random_times - 1

				node.connections[neighbour_index] = 1
				self.nodes[neighbour_index].connections[index] = 1

	def plot(self):
		# given code
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_axis_off()

			num_nodes = len(self.nodes)
			network_radius = num_nodes * 10
			ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
			ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

			for (i, node) in enumerate(self.nodes):
				node_angle = i * 2 * np.pi / num_nodes
				node_x = network_radius * np.cos(node_angle)
				node_y = network_radius * np.sin(node_angle)

				circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
				ax.add_patch(circle)

				for neighbour_index in range(i + 1, num_nodes):
					if node.connections[neighbour_index]:
						neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
						neighbour_x = network_radius * np.cos(neighbour_angle)
						neighbour_y = network_radius * np.sin(neighbour_angle)

						ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment - Unfinished
==============================================================================================================
'''
def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''
	row, col = population.shape

	def population(row, col):
		return np.random.choice([-1, 1], size=(row, col))

	def Change_in_agreement(population, row, col, external=0.0):
		total = 0
		for i in range(row - 1, row + 2):
			for j in range(col - 1, col + 2):
				if i == row and j == col:
					continue
				elif i == row - 1 and j == col - 1:
					continue
				elif i == row - 1 and j == col + 1:
					continue
				elif i == row + 1 and j == col - 1:
					continue
				elif i == row + 1 and j == col + 1:
					continue
				total += population(row, col)
		Change_in_agreement = population(row, col) * total

		return np.random.random() * population



def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1
	elif agreement > 0:
		population[row, col] == population[row, col]

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)



def ising_main(population, alpha=None, external=0.0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)

'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 & 5 in the assignment
==============================================================================================================
'''
def plot_hist(op = [], ttl = None):
	plt.figure(figsize=(8, 4))
	plt.hist(np.transpose(op), edgecolor='black')
	plt.title(ttl)
	plt.xlabel('Opinion')

def get_opinions(nw):
	opinions = [0 for _ in range(len(nw.nodes))]
	for node in nw.nodes:
		opinions[node.index] = node.value
	return opinions

def get_mean(opinions_history):
	mean_op = [0 for _ in range(len(opinions_history[0]))]
	for opinions in opinions_history:
		for i in range(len(opinions)):
			mean_op[i] += opinions[i]
	for j in range(len(mean_op)):
		mean_op[j] /= len(opinions_history)
	return mean_op

def defuant_main(num_people = 50, threshold = 0.2, beta = 0.2, nw = None):
	
	if nw is None:	
		# initial opinions
		opinions = np.random.rand(num_people)

		plot_hist(opinions,'Initial Opinions')
		plt.show()

		# Store opinion data after each iteration
		opinions_history = [opinions.copy()]

		# Simulation opinion update
		for each in range(num_people*100):
			# Randomly choose a person
			person_idx = np.random.randint(0, num_people)
		
			# Randomly select your left or right neighbors
			direction = np.random.choice([-1, 1])
			neighbor_idx = (person_idx + direction) % num_people
		
			# Check for differences in opinions
			diff = abs(opinions[person_idx] - opinions[neighbor_idx])
			
			if diff < threshold:
				# update opinions
				opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
				opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
			# Store current opinion data
			opinions_history.append(opinions.copy())

			# Visualize changes in opinion during iteration
		plt.figure(figsize=(10, 6))
		for i, opinion in enumerate(opinions_history):
			plt.plot([i]*num_people, opinion, 'o', markersize=2, color='blue', alpha=0.5)

		plt.title('Opinions Evolution During Iterations')
		plt.xlabel('Iteration')
		plt.ylabel('Opinion')
		plt.show()

		# Visualize the updated comments
		plot_hist(opinions,'Final Opinions')
		plt.show()
	else:
		# A Defuant model implemented using Network
		node_history = []
		for node in nw.nodes:
			node_history.append(node)
			neighbours = [nw.nodes[neighbour] for neighbour, connection in enumerate(node.connections) if connection]
			for neighbour in neighbours:
				node.update_value(neighbour.value, threshold, beta)
				neighbour.update_value(node.value, threshold, beta)
				#print('updating values')

		nw.update_nodes(node_history)
		return nw
	
'''
==============================================================================================================
This section contains code for the Networks and Small world networks - task 3 & 4 in the assignment
==============================================================================================================
'''

def search_paths(network, start_node): #apply breadth-first-search
	#####task 3#####
	#to search paths to all other nodes, return the distance to each other nodes

	paths = {}
	queue = Queue()
	queue.push((start_node,0))
	visited = []
	visited.append(start_node.index) #add the node being check into the visited list so it will not count itself and
									#the path to itself

	while not queue.is_empty():
		start_node, distance = queue.pop()

		for neighbour_index, connection in enumerate(start_node.connections):
			if connection and neighbour_index not in visited:
				neighbour = network.nodes[neighbour_index]
				paths[neighbour.index] = distance + 1
				queue.push((neighbour, distance + 1))
				visited.append(neighbour_index)

	return paths


'''
==============================================================================================================
5 - This section contains code for ALL TEST FUNCTIONS
==============================================================================================================
'''
def test_ising():

	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==14), "Test 9"
	assert(calculate_agreement(population,1,1, -10)==-6), "Test 10"

	print("Tests passed")

def test_defuant(num_people = 50, threshold = 0.2, beta = 0.2):
	
	print(f'total number of people in this test function is {num_people}')
	print(f'threshold set at {threshold}')
	print(f'beta set at {beta}')

	# initial opinions
	opinions = np.random.rand(num_people)

	plot_hist(opinions,'Initial Opinions')
	plt.show()

	# update opinions
	for each in range(3):
		# randomly choose a person
		person_idx = np.random.randint(0, num_people)
		print(f'------iteration {each}------')
		print(f'people No. {person_idx} chosen as Xi')
		
		# randomly choose left neighbour or right neighbour
		direction = np.random.choice([-1, 1])
		print(f'direction {direction} chosen')
		neighbor_idx = (person_idx + direction) % num_people
		print(f'people No. {neighbor_idx} chosen as Xj')
		
		# Check for difference in opinions
		diff = abs(opinions[person_idx] - opinions[neighbor_idx])
		
		if diff < threshold:
			# update opinions
			print('diff < threshold, updating opinions')
			print(f'Xi({each}) = {opinions[person_idx]}, Xj({each}) = {opinions[neighbor_idx]}')
			opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
			opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
			print(f'Xi({each+1}) = {opinions[person_idx]}, Xj({each+1}) = {opinions[neighbor_idx]}')
		else:
			print('diff >= threshold, no changes made')

	# Visualize the updated comments
	plot_hist(opinions,'Final Opinions')
	plt.show()

def test_network():
	#some function names are changed due to consistency

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
	assert(network.get_mean_clustering()==0), network.get_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_path_length()

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
	assert(network.get_mean_clustering()==0),  network.get_clustering()
	assert(network.get_mean_path_length()==5), network.get_path_length()

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
	assert(network.get_mean_clustering()==1),  network.get_clustering()
	assert(network.get_mean_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
6 - This section contains code for FLAGS
==============================================================================================================
'''
def parse(arg):
	######task 2 & 5######
	defuant = 0
	threshold = 0
	beta = 0
	test_defuant = 0

	use_network = 0
	num_people = 0

	if '-defuant' in sys.argv:
		defuant = 1
		
		if '-threshold' in sys.argv:
			threshold = 1

		if '-beta' in sys.argv:
			beta = 1

		if '-use_network' in sys.argv:
			use_network = 1

	if '-test_defuant' in sys.argv:
		test_defuant = 1

	if '-num_people' in sys.argv:
		num_people = 1
		
	######task 3######
	network = 0
	test_network = 0

	if '-network' in sys.argv:
		network = 1

	if '-test_network' in sys.argv:
		test_network = 1

	######task 3######
	ring_network = 0
	small_world = 0
	re_wire = 0
	
	if '-ring_network' in sys.argv:
		ring_network = 1

	if '-small_world' in sys.argv:
		small_world = 1

	if '-re_wire' in sys.argv:
		rewire = 1

	return defuant, threshold, beta, test_defuant, use_network, num_people, network, test_network, ring_network, small_world, re_wire
	
def flags():
	flag = argparse.ArgumentParser(description="type your flags")

	#####task 2 & 5#####
	flag.add_argument('-defuant')
	flag.add_argument('beta')
	flag.add_argument('-threshold')
	flag.add_argument('-test_defuant')

	flag.add_argument('-use_network')
	flag.add_argument('-num_people')
	
	#####task 3#####
	flag.add_argument('-network', action='store_true')
	flag.add_argument('-test_network', action='store_true')

	#####task 4#####
	flag.add_argument("-ring_network", "--ring_network", help = "Input the Network Size")
	flag.add_argument("-small_world", "--small_world", help = "Input the Network Size")
	flag.add_argument("-re_wire", "--re_wire", help = "Input the re-wiring probability of your small worlds network", default="0.2")
	
'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main(args):
	#take flag values
	fdefuant, fthreshold, fbeta, ftest_defuant, fuse_network, fnum_people, fnetwork, ftest_network, fring_network, fsmall_world, fre_wire = parse(args)
	
	###default values for task 2###
	num_people = 50
	threshold = 0.2
	beta = 0.2
	
	###default values for task 3###
	connectivity_p = 0.5

	###default values for task 4###
	re_wire_prob = 0.2

	#####TESTS#####
	if ftest_defuant:
		test_defuant(num_people, threshold, beta)

	if ftest_network:
		test_network()

	#####SEPARATED PARTS#####

	#####Task 2 & 5#####
	if fdefuant:
		if fthreshold:
			threshold_idx = sys.argv.index('-threshold')
			threshold = float(sys.argv[threshold_idx + 1])
		if fbeta:
			beta_idx = sys.argv.index('-beta')
			beta = float(sys.argv[beta_idx + 1])
		if fnum_people:
			num_people_idx = sys.argv.index('-num_people')
			num_people = int(sys.argv[num_people_idx + 1])

		#####Task 5#####
		if fuse_network:
			network_size_idx = sys.argv.index('-use_network')
			network_size = int(sys.argv[network_size_idx+1])
			nw = Network()
			Network.make_random_network(nw, network_size, connectivity_p)
					
			print("Initial Network:")
			nw.plot()
			opinions = get_opinions(nw)
			opinions_history = [opinions.copy()]
			plot_hist(opinions, 'Initial Opinions:')
			plt.show()

			for i in range(5):
				print(f"Iteration {i+1}:")
				nw = defuant_main(network_size, threshold, beta, nw)
				opinions = get_opinions(nw)
				opinions_history.append(opinions)

			def update(frame):
				opinions = opinions_history[frame]
				plt.hist(np.transpose(opinions), edgecolor='black')
				return opinions

			fig, ax = plt.subplots()
			# Create animations in frames as the number of iterations, with 100 milliseconds between each frame
			ani = animation.FuncAnimation(fig, update, frames=range(len(opinions_history)), interval=100, repeat=False)
			plt.show()
					
			plot_hist(get_mean(opinions_history), 'Mean Opinions:')
			plt.show()
					
		else:
			defuant_main(num_people, threshold, beta)
			
	#####Task 3#####
	if fnetwork:
		network_idx = sys.argv.index('-network')
		network_size = int(sys.argv[network_idx + 1])
		
		Network.make_random_network(Network, network_size, connectivity_p)
		print('Mean degree:', Network.get_mean_degree(Network))
		print('Average path length:', Network.get_mean_path_length(Network))
		print('Clustering coefficient:', Network.get_mean_clustering(Network))
		Network.plot(Network)
		plt.show()

	#####Task 4#####	
	if fring_network:

		ring_network_size_idx = sys.argv.index('-ring_network')
		ring_network_size_idx = int(sys.argv[ring_network_size_idx + 1])

		# print("ring_network", args.ring_network)
		n = Network()
		n.make_ring_network(int(ring_network_size_idx))
		n.plot()
		plt.show()

	if fsmall_world:

		small_world_size_idx = sys.argv.index('-small_world')
		small_world_size = int(sys.argv[small_world_size_idx + 1])

		if fre_wire:
			re_wire_idx = sys.argv.index('-re_wire')
			re_wire_prob = int(sys.argv[re_wire_idx + 1])

		# print("small_world", args.small_world)
		n = Network()
		n.make_small_world_network(small_world_size, re_wire_prob)
		n.plot()
		plt.show()
			
if __name__=="__main__":
	main(sys.argv[1:])


