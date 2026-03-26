import numpy as np

class Tree:
	def __init__(self, root):
		self.root = root
		self.nodes = []
		self.build_tree_from_root()

	def get_node_of_state(self, state: np.array):
		for node in self.nodes:
			if np.array_equal(node.state, state):
				return node
		return None

	def add_node(self, node):
		# WARN: this implementation expects no duplicates are added; we do not enforce this
		self.nodes.append(node)

	def build_tree_from_root(self):
		""" Traverse the tree using BFS and add nodes
		to self.nodes"""
		queue = [self.root]
		while len(queue) > 0:
			current = queue.pop()
			if self.get_node_of_state(current) is None:
				self.add_node(current)
			if not current.is_leaf():
				for edge in current.out_edges:
					# Only has destination node if it has been traversed at least once
					if edge.n > 0:
						queue.insert(0, edge.to)
		

class Node:
	def __init__(self, env, state: np.array, in_edge = None):
		self.env = env
		self.state = state
		self.action_mask = self.env.get_action_mask(self.state)
		self.in_edge = in_edge
		self.valid_actions = None
		self.priors = None
		self.out_edges = None
		# We compute and store Q(s, a) + U(s, a) values associated with each outgoing edge;
		# we do it here to optimise search speed and space-complexity
		self.qu_values = None 

	def __eq__(self, node):
		return np.array_equal(self.state, node.state)

	def save_in_edge(self, edge):
		# This is used to save the path we've taken in a particular simulation
		# Several incoming edges exist in theory
		self.in_edge = edge

	def expand(self, priors: np.array):
		"""
		Compute valid actions,
		Set priors,
		Construct edges.
		"""
		if self.env.is_terminal(self.state):
			# If state is terminal, there's noting to expand
			pass
		else:
			self.valid_actions = np.stack(self.action_mask.nonzero()).T
			self.priors = priors[np.nonzero(self.action_mask.reshape(-1))]
			self.out_edges = [
				Edge(
					env=self.env, frm=self, action=action
				) for i, action in enumerate(self.valid_actions)
			]

	def get_out_edge_by_action(self, action: np.array):
		for edge in self.out_edges:
			if np.array_equal(edge.action, action):
				return edge
		return None

	def is_leaf(self):
		return self.out_edges is None
	
	def is_terminal(self):
		return self.env.is_terminal(self.state)
	
	def update_qu_values(self, c: float):
		"""
		Compute Q(s, a) + U(s, a) for each outgoing edge.
		"""
		# WARN: the paper is ambigous here.
		# In my reading, U(s, a) = 0 if all other
		# edges at the same level are unvisited.
		# For now, we stick to the formula in the paper,
		# but we may want to consider selecting values based on the prior
		if self.is_leaf():
			raise Exception("U-values cannot be computed for leaf node.")
		sum_n_sb = 0
		for edge in self.out_edges:
			sum_n_sb += edge.n
		sqrt_sum_n_sb = np.sqrt(sum_n_sb)
		# U(s, a) = c*P*(sqrt(sum_b N(s, b))/ 1 + N(s, a))
		self.qu_values = [edge.q + (c * self.priors[i] * (sqrt_sum_n_sb / (1 + edge.n))) for i, edge in enumerate(self.out_edges)]
		self.qu_values = np.array(self.qu_values)
	
	def get_pi_values(self, tau: float):
		if self.is_leaf():
			raise Exception("Search probabilities cannot be computed for leaf node.")
		sum_exp_nbs = 0
		for edge in self.out_edges:
			sum_exp_nbs += edge.n ** (1 / tau)
		pi_values = [(edge.n ** (1 / tau)) / sum_exp_nbs for edge in self.out_edges]
		return np.array(pi_values)
	




class Edge:
	def __init__(self, env, frm: Node, action: np.array):
		self.env = env
		self.frm: Node = frm
		self.action = action
		self.to = None
			
		self.n = 0
		self.w = 0
		self.q = 0

	def __eq__(self, edge):
		return self.frm == edge.frm and self.to == edge.to
	
	def add_destination_node(self, tree: Tree):
		# Search if the node already exists in the tree
		# We avoid duplicating nodes
		to_state = self.env.simulate_step(self.frm.state, self.action)[0]
		self.to = tree.get_node_of_state(to_state)
		if self.to is None:
			self.to: Node = Node(
				env=self.env,
				state=to_state,
				in_edge=self,
			)
			# The node is new, so add it to the tree
			tree.add_node(self.to)

	def update(self, v: float):
		self.n += 1
		self.w += v
		self.q = self.w / self.n
