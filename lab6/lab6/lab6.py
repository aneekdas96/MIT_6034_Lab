# MIT 6.034 Lab 6: Neural Nets

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
    	return 1
    else:
    	return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    exp_den = e**( (-(steepness)) * (x - midpoint))
    den = 1 + exp_den
    return 1/(den)

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0, x)

    
# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return float(-0.5 * (desired_output - actual_output)**2)



#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    # sample : ['line1', 'line2', 'X1', 'X2', AND]
    neuron_outputs = {}
    for neuron in net.neurons:
    	neuron_outputs[neuron] = 0
    for node in net.topological_sort():
    	# if node in net.inputs:
    	# 	pass
    	if net.is_output_neuron(node):
    		connections = net.get_wires(startNode = None, endNode = node)
    		total_val = 0
    		for edge in connections:
    			start_node = edge.startNode
    			node_output = node_value(start_node, input_values, neuron_outputs)
    			weight = edge.get_weight()
    			prod = node_output * weight 
    			total_val += prod
    		activation = threshold_fn(total_val)
    		neuron_outputs[node] = activation
    		return activation, neuron_outputs
    	else:
    		connections = net.get_wires(startNode = None, endNode = node)
    		total_val = 0
    		for edge in connections:
    			start_node = edge.startNode
    			node_output = node_value(start_node, input_values, neuron_outputs)
    			weight = edge.get_weight()
    			prod = node_output * weight 
    			total_val += prod
    		activation = threshold_fn(total_val)
    		neuron_outputs[node] = activation

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    max_val = -INF
    max_inp = None
    possible_operations = [step_size, -step_size, 0]
    possible_combinations = []
    for operation1 in possible_operations:
    	for operation2 in possible_operations:
    		for operation3 in possible_operations:
    			possible_combinations.append([operation1, operation2, operation3])
    for i in range(len(possible_combinations)):
    	current_combination = possible_combinations[i]
    	temp = [inputs[0] + current_combination[0], inputs[1] + current_combination[1], inputs[2] + current_combination[2]]
    	func_val = func(temp[0], temp[1], temp[2])
    	if func_val > max_val:
    		max_val = func_val
    		max_inp = temp

    return max_val, max_inp

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    set_of_dependencies = set()
    agenda = []
    agenda.append(wire)
    while agenda != []:	
    	wire = agenda.pop(0)
    	set_of_dependencies.add(wire)
    	start_node = wire.startNode 
    	end_node = wire.endNode
    	set_of_dependencies.add(start_node)
    	set_of_dependencies.add(end_node)
    	next_wires = net.get_wires(startNode = end_node)
    	agenda = next_wires + agenda

    return set_of_dependencies

def delta(neuron, net, desired_output, neuron_outputs):
	n_out = neuron_outputs[neuron]
	if net.is_output_neuron(neuron):
		update_coeff = n_out * (1 - n_out) * (desired_output - n_out)
	else:
		total = 0
		neighbor_wires = net.get_wires(startNode = neuron)
		for wire in neighbor_wires:
			end_node_wire = wire.endNode 
			total += (wire.get_weight() * delta(end_node_wire, net, desired_output, neuron_outputs))
		update_coeff = n_out * (1 - n_out) * total 
	return update_coeff

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    delta_dict = {}
    for neuron in net.topological_sort():
    	update_coeff = delta(neuron, net, desired_output, neuron_outputs)
    	delta_dict[neuron] = update_coeff
    return delta_dict
   
def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    wires = net.get_wires()
    delta_values = calculate_deltas(net, desired_output, neuron_outputs)
    for wire in wires:
    	start_node = wire.startNode 
    	end_node = wire.endNode 
    	if start_node in input_values.keys():
    		del_w = r * input_values[start_node] * delta_values[end_node]
    	elif start_node in neuron_outputs.keys():
    		del_w = r * neuron_outputs[start_node] * delta_values[end_node]
    	else:
    		del_w = r * start_node * delta_values[end_node]
    	new_weight = wire.get_weight() + del_w
    	wire.set_weight(new_weight)
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    epochs = 0
    neuron_outputs = forward_prop(net, input_values, threshold_fn = sigmoid)
    actual_output = neuron_outputs[0]
    net_accuracy = accuracy(desired_output, actual_output)
    while net_accuracy < minimum_accuracy:
    	net = update_weights(net, input_values, desired_output, neuron_outputs[1], r)
    	neuron_outputs = forward_prop(net, input_values, threshold_fn=sigmoid)
    	actual_output = neuron_outputs[0]
    	net_accuracy = accuracy(desired_output, actual_output)
    	epochs += 1
    return net, epochs


#### Part 5: Training a Neural Net #############################################
# Answers are obtained from running the neural net.

ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A', 'C']
ANSWER_12 = ['A','E']


#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
