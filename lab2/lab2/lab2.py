# MIT 6.034 Lab 2: Search
# Written by 6.034 staff

from search import Edge, UndirectedGraph, do_nothing_fn, make_generic_search
import read_graphs
from functools import reduce

all_graphs = read_graphs.get_graphs()
GRAPH_0 = all_graphs['GRAPH_0']
GRAPH_1 = all_graphs['GRAPH_1']
GRAPH_2 = all_graphs['GRAPH_2']
GRAPH_3 = all_graphs['GRAPH_3']
GRAPH_FOR_HEURISTICS = all_graphs['GRAPH_FOR_HEURISTICS']


# Please see wiki lab page for full description of functions and API.

#### PART 1: Helper Functions ##################################################

def path_length(graph, path):
    """Returns the total length (sum of edge weights) of a path defined by a
    list of nodes coercing an edge-linked traversal through a graph.
    (That is, the list of nodes defines a path through the graph.)
    A path with fewer than 2 nodes should have length of 0.
    You can assume that all edges along the path have a valid numeric weight."""
    graph = graph 
    path = path 
    tot_length = 0
    for i in range(len(path)-1):
    	ini_node = path[i]
    	fin_node = path[i+1]
    	edge = graph.get_edge(ini_node, fin_node)
    	if edge:
    		length_of_edge = edge.length
    		if length_of_edge != None:
    			tot_length += length_of_edge
    return tot_length

def has_loops(path):
    """Returns True if this path has a loop in it, i.e. if it
    visits a node more than once. Returns False otherwise."""
    flag = 0
    for node in path:
    	if path.count(node) > 1:
    		flag = 1
    if flag == 1:
    	return True
    else:
    	return False

def create_new_path(path, nbor):
	temp = []
	for node in path:
		temp.append(node)
	temp.append(nbor)
	return temp

def extensions(graph, path):
    """Returns a list of paths. Each path in the list should be a one-node
    extension of the input path, where an extension is defined as a path formed
    by adding a neighbor node (of the final node in the path) to the path.
    Returned paths should not have loops, i.e. should not visit the same node
    twice. The returned paths should be sorted in lexicographic order."""
    fin_node = path[-1]
    list_of_neighbors = sorted(graph.get_neighbors(fin_node))
    list_of_paths = []
    for node in list_of_neighbors:
    	node_path = create_new_path(path, node)
    	if has_loops(node_path) == False:
    		list_of_paths.append(node_path)
    return list_of_paths

def sort_by_heuristic(graph, goalNode, nodes):
    """Given a list of nodes, sorts them best-to-worst based on the heuristic
    from each node to the goal node. Here, and in general for this lab, we
    consider a smaller heuristic value to be "better" because it represents a
    shorter potential path to the goal. Break ties lexicographically by 
    node name."""
    nodes = sorted(nodes, key = lambda node: (graph.get_heuristic_value(node, goalNode), node))
    return nodes

# You can ignore the following line.  It allows generic_search (PART 3) to
# access the extensions and has_loops functions that you just defined in PART 1.
generic_search = make_generic_search(extensions, has_loops)  # DO NOT CHANGE


#### PART 2: Basic Search ######################################################

################################DFS#################################

def ret_dfs(graph, agenda, goal):
	if agenda!= []:
		curr_path = agenda[0]
		if goal in curr_path:
			return curr_path
		else:
			list_of_extensions = extensions(graph, curr_path)
			agenda.pop(0)
			for i in range(len(list_of_extensions)-1, -1, -1):
				agenda.insert(0, list_of_extensions[i])
			return ret_dfs(graph, agenda, goal)
	else:
		return None


def basic_dfs(graph, start, goal):
    """
    Performs a depth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    Uses backtracking, but does not use an extended set.
    """
    agenda = []
    agenda.append([start])
    store = ret_dfs(graph, agenda, goal)
    return store

################################BFS#################################

def ret_bfs(graph, agenda, goal):
	if agenda!=[]:
		current_path = agenda[0]
		if goal in current_path:
			return current_path
		else:
			extended_paths = extensions(graph, current_path)
			agenda.pop(0)
			for path in extended_paths:
				agenda.append(path)
			return ret_bfs(graph, agenda, goal)
	else:
		return None

def basic_bfs(graph, start, goal):
    """
    Performs a breadth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    """
    agenda = []
    agenda.append([start])
    store = ret_bfs(graph, agenda, goal)
    return store

#### PART 3: Generic Search ####################################################

# Generic search requires four arguments (see wiki for more details):
# sort_new_paths_fn: a function that sorts new paths that are added to the agenda
# add_paths_to_front_of_agenda: True if new paths should be added to the front of the agenda
# sort_agenda_fn: function to sort the agenda after adding all new paths 
# use_extended_set: True if the algorithm should utilize an extended set


# Define your custom path-sorting functions here.
# Each path-sorting function should be in this form:
# def my_sorting_fn(graph, goalNode, paths):
#     # YOUR CODE HERE
#     return sorted_paths

def lexical_sort(graph, goalNode, paths):
	paths = sorted(paths, key = lambda path:path[-1])
	return paths

def heuristic_sort(graph, goalNode, paths):
	paths = sorted(paths, key = lambda path:path[-1])
	paths = sorted(paths, key = lambda path: graph.get_heuristic_value(path[-1], goalNode))
	return paths 

def total_length_sort(graph, goalNode, paths):
	paths = sorted(paths, key = lambda path:path_length(graph, path))
	return paths

def get_total_length_with_heuristic(graph, goalNode, path):
	length_of_path = path_length(graph, path)
	heuristic_length = graph.get_heuristic_value(path[-1], goalNode)
	return length_of_path + heuristic_length

def total_length_with_heuristic_sort(graph, goalNode, paths):
	paths = sorted(paths, key = lambda path:(get_total_length_with_heuristic(graph, goalNode, path), path[-1]))
	return paths

generic_dfs = [lexical_sort, True, do_nothing_fn, False]

generic_bfs = [lexical_sort, False, do_nothing_fn, False]

generic_hill_climbing = [heuristic_sort, True, do_nothing_fn, False]

generic_best_first = [do_nothing_fn, False, heuristic_sort, False]

generic_branch_and_bound = [do_nothing_fn, False, total_length_sort, False]

generic_branch_and_bound_with_heuristic = [do_nothing_fn, False, total_length_with_heuristic_sort, False]

generic_branch_and_bound_with_extended_set = [do_nothing_fn, False, total_length_sort, True]

generic_a_star = [do_nothing_fn, False, total_length_with_heuristic_sort, True]


# Here is an example of how to call generic_search (uncomment to run):
# my_dfs_fn = generic_search(*generic_dfs)
# my_dfs_path = my_dfs_fn(GRAPH_2, 'S', 'G')
# print(my_dfs_path)

# Or, combining the first two steps:
# my_dfs_path = generic_search(*generic_dfs)(GRAPH_2, 'S', 'G')
# print(my_dfs_path)


### OPTIONAL: Generic Beam Search

# If you want to run local tests for generic_beam, change TEST_GENERIC_BEAM to True:
TEST_GENERIC_BEAM = False

# The sort_agenda_fn for beam search takes fourth argument, beam_width:
# def my_beam_sorting_fn(graph, goalNode, paths, beam_width):
#     # YOUR CODE HERE
#     return sorted_beam_agenda

generic_beam = [None, None, None, None]


# Uncomment this to test your generic_beam search:
# print(generic_search(*generic_beam)(GRAPH_2, 'S', 'G', beam_width=2))


#### PART 4: Heuristics ########################################################

def is_admissible(graph, goalNode):
    """Returns True if this graph's heuristic is admissible; else False.
    A heuristic is admissible if it is either always exactly correct or overly
    optimistic; it never over-estimates the cost to the goal."""
    flag = 0
    nodes = graph.nodes
    for node in nodes:
    	shortest_path = basic_bfs(graph, node, goalNode)
    	shortest_path_length = path_length(graph, shortest_path)
    	heuristic_length =  graph.get_heuristic_value(node, goalNode)
    	if heuristic_length <= shortest_path_length:
    		pass
    	else:
    		flag = 1
    if flag == 1:
    	return False
    else:
    	return True


def is_consistent(graph, goalNode):
    """Returns True if this graph's heuristic is consistent; else False.
    A consistent heuristic satisfies the following property for all
    nodes v in the graph:
        Suppose v is a node in the graph, and N is a neighbor of v,
        then, heuristic(v) <= heuristic(N) + edge_weight(v, N)
    In other words, moving from one node to a neighboring node never unfairly
    decreases the heuristic.
    This is equivalent to the heuristic satisfying the triangle inequality."""
    flag = 0
    edges_in_graph = graph.edges
    for edge in edges_in_graph:
    	start = edge.startNode
    	end = edge.endNode
    	length = edge.length
    	if length >= abs(graph.get_heuristic_value(start, goalNode) - graph.get_heuristic_value(end, goalNode)):
    		pass
    	else:
    		flag = 1
    if flag == 1:
    	return False
    else:
    	return True



### OPTIONAL: Picking Heuristics

# If you want to run local tests on your heuristics, change TEST_HEURISTICS to True.
#  Note that you MUST have completed generic a_star in order to do this:
TEST_HEURISTICS = False


# heuristic_1: admissible and consistent

[h1_S, h1_A, h1_B, h1_C, h1_G] = [None, None, None, None, None]

heuristic_1 = {'G': {}}
heuristic_1['G']['S'] = h1_S
heuristic_1['G']['A'] = h1_A
heuristic_1['G']['B'] = h1_B
heuristic_1['G']['C'] = h1_C
heuristic_1['G']['G'] = h1_G


# heuristic_2: admissible but NOT consistent

[h2_S, h2_A, h2_B, h2_C, h2_G] = [None, None, None, None, None]

heuristic_2 = {'G': {}}
heuristic_2['G']['S'] = h2_S
heuristic_2['G']['A'] = h2_A
heuristic_2['G']['B'] = h2_B
heuristic_2['G']['C'] = h2_C
heuristic_2['G']['G'] = h2_G


# heuristic_3: admissible but A* returns non-optimal path to G

[h3_S, h3_A, h3_B, h3_C, h3_G] = [None, None, None, None, None]

heuristic_3 = {'G': {}}
heuristic_3['G']['S'] = h3_S
heuristic_3['G']['A'] = h3_A
heuristic_3['G']['B'] = h3_B
heuristic_3['G']['C'] = h3_C
heuristic_3['G']['G'] = h3_G


# heuristic_4: admissible but not consistent, yet A* finds optimal path

[h4_S, h4_A, h4_B, h4_C, h4_G] = [None, None, None, None, None]

heuristic_4 = {'G': {}}
heuristic_4['G']['S'] = h4_S
heuristic_4['G']['A'] = h4_A
heuristic_4['G']['B'] = h4_B
heuristic_4['G']['C'] = h4_C
heuristic_4['G']['G'] = h4_G


##### PART 5: Multiple Choice ##################################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '1'

ANSWER_4 = '3'


#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''



###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the online tester. DO NOT CHANGE!

generic_dfs_sort_new_paths_fn = generic_dfs[0]
generic_bfs_sort_new_paths_fn = generic_bfs[0]
generic_hill_climbing_sort_new_paths_fn = generic_hill_climbing[0]
generic_best_first_sort_new_paths_fn = generic_best_first[0]
generic_branch_and_bound_sort_new_paths_fn = generic_branch_and_bound[0]
generic_branch_and_bound_with_heuristic_sort_new_paths_fn = generic_branch_and_bound_with_heuristic[0]
generic_branch_and_bound_with_extended_set_sort_new_paths_fn = generic_branch_and_bound_with_extended_set[0]
generic_a_star_sort_new_paths_fn = generic_a_star[0]

generic_dfs_sort_agenda_fn = generic_dfs[2]
generic_bfs_sort_agenda_fn = generic_bfs[2]
generic_hill_climbing_sort_agenda_fn = generic_hill_climbing[2]
generic_best_first_sort_agenda_fn = generic_best_first[2]
generic_branch_and_bound_sort_agenda_fn = generic_branch_and_bound[2]
generic_branch_and_bound_with_heuristic_sort_agenda_fn = generic_branch_and_bound_with_heuristic[2]
generic_branch_and_bound_with_extended_set_sort_agenda_fn = generic_branch_and_bound_with_extended_set[2]
generic_a_star_sort_agenda_fn = generic_a_star[2]

# Creates the beam search using generic beam args, for optional beam tests
beam = generic_search(*generic_beam) if TEST_GENERIC_BEAM else None

# Creates the A* algorithm for use in testing the optional heuristics
if TEST_HEURISTICS:
    a_star = generic_search(*generic_a_star)
