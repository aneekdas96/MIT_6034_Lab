# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    flag = 0
    for i in csp.domains:
    	if len(csp.domains[i]) == 0:
    		flag = 1
    if flag == 0:
    	return False
    else:
    	return True

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    flag = 0
    assignments = csp.assignments
    for assignment in assignments:
    	constraints = csp.constraints_between(assignment, None)
    	for constraint in constraints:
    		if constraint.var2 in assignments:
    			var1 = constraint.var1
    			var2 = constraint.var2
    			val1 = assignments[var1]
    			val2 = assignments[var2]
    			if constraint.check(val1, val2) == False:
    				flag = 1
    		else:
    			pass
    if flag == 1:
    	return False
    else:
    	return True 


#### Part 2: Depth-First Constraint Solver #####################################

def create_extensions(problem, variable):
	list_of_extensions = []
	list_of_values = problem.get_domain(variable)
	if list_of_values != []:
		for value in list_of_values:
			new_problem = problem.copy()
			new_problem.set_assignment(variable, value)
			list_of_extensions.append(new_problem)
	return list_of_extensions

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = []
    extensions = 0
    agenda.append(problem)
    while agenda != []:
        curr_problem = agenda[0]
        agenda.pop(0)
        extensions += 1
        if check_all_constraints(curr_problem) == True and has_empty_domains(curr_problem) == False:
            next_unassigned_var = curr_problem.pop_next_unassigned_var()
            if next_unassigned_var == None:
                return curr_problem.assignments, extensions
            else:
                list_of_extensions = create_extensions(curr_problem, next_unassigned_var)
                agenda = list_of_extensions + agenda
    if agenda == []:
        return None, extensions
    

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?
pokemon_problem = get_pokemon_problem()
solution, extensions = solve_constraint_dfs(pokemon_problem)
# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.
ANSWER_1 = extensions


#### Part 3: Forward Checking ##################################################

def helper_checking_neighbors(csp, constraints, val_neighbor, values_for_var):
    flag = [True] * len(values_for_var)
    for i, val in enumerate(values_for_var):
        for constraint in constraints:
            if constraint.check(val, val_neighbor) == False:
                flag[i] = False
    if True in flag:
        return False
    else:
        return True

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    values_to_delete = []
    reduced_list = []
    values_for_var = csp.get_domain(var)
    neighbors = csp.get_neighbors(var)
    for neighbor in neighbors:
        values_for_neighbor = csp.get_domain(neighbor)
        constraints = csp.constraints_between(var, neighbor)
        values_to_delete = []
        for val_neighbor in values_for_neighbor:
            should_remove_val_neighbor = helper_checking_neighbors(csp, constraints, val_neighbor, values_for_var)
            if should_remove_val_neighbor == True:
                values_to_delete.append(val_neighbor)
                if neighbor not in reduced_list:
                    reduced_list.append(neighbor)
        for value in values_to_delete:
            csp.domains[neighbor].remove(value)
        if has_empty_domains(csp) == True:
            return None
    return sorted(reduced_list)

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def create_extensions_with_forward_check(problem, variable):
	list_of_extensions = []
	list_of_values = problem.get_domain(variable)
	if list_of_values != []:
		for value in list_of_values:
			new_problem = problem.copy()
			new_problem = new_problem.set_assignment(variable, value)####For convenience, also modifies the variable's domain to contain only the assigned value.
			res = eliminate_from_neighbors(new_problem, variable)
			list_of_extensions.append(new_problem)
	return list_of_extensions

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = []
    extensions = 0
    agenda.append(problem)
    while agenda != []:
        curr_problem = agenda[0]
        agenda.pop(0)
        extensions += 1
        #if it does not have empty domain and u meet all the constraints
        # then check free var 
        # if no free var hten done 
        # otherweise extend  problem set 
        # add new prob to agenda
        if has_empty_domains(curr_problem) == False and check_all_constraints(curr_problem) == True:
            next_unassigned_var = curr_problem.pop_next_unassigned_var()
            if next_unassigned_var == None:
                return curr_problem.assignments, extensions
            else:
                list_of_extensions = create_extensions_with_forward_check(curr_problem, next_unassigned_var)
                for i in range(len(list_of_extensions)-1, -1, -1):
                    agenda.insert(0, list_of_extensions[i])
    else:
        return None, extensions


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

pokemon_problem = get_pokemon_problem()
solution, extensions = solve_constraint_forward_checking(pokemon_problem)
ANSWER_2 = extensions


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    #establish a queue
    #if using domain reduction during search, queue should only contain the variable that was assigned
    #until the queue is empty, pop the first variable off the queue
    #iterate over the variables neighbors
    #if some neighbor n has values that are inconsistent between var and n, remove inconsistent values from n's domain
    #if you reduce a neighbors domain, add that neighbor to the queue  (unless it's already in the queue)
    #if any variable has an empty domain, quit immediately and return None
    #when the queue is empty, domain reduction has been finished
    #return a list of all variables that have been dequed, inorder they were removd from the queue.
    dequed_list = []
    csp_copy = csp.copy()
    if queue == None:
        queue = csp_copy.get_all_variables()
    while queue != []:
        var = queue.pop(0)
        dequed_list.append(var)
        reduced_list = eliminate_from_neighbors(csp, var)
        if reduced_list == None:
            return None
        elif reduced_list == []:
            pass
        elif reduced_list != [] and reduced_list != None:
            for neighbor in reduced_list:
                if neighbor not in queue:
                    queue.append(neighbor)
            if has_empty_domains(csp) == True:
                return None 
    if queue == []:
        return dequed_list

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

pokemon_problem = get_pokemon_problem()
domain_reduction(pokemon_problem)
solution_1, extension_1 = solve_constraint_dfs(pokemon_problem)
ANSWER_3 = extension_1

def create_extensions_with_domain_reduction(problem, variable):
    list_of_extensions = []
    list_of_values = problem.get_domain(variable)
    if list_of_values != []:
        for value in list_of_values:
            new_problem = problem.copy()
            new_problem = new_problem.set_assignment(variable, value)####For convenience, also modifies the variable's domain to contain only the assigned value.
            queue = [variable]
            res = domain_reduction(new_problem, queue)
            list_of_extensions.append(new_problem)
    return list_of_extensions

def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = []
    agenda.append(problem)
    extensions = 0
    while agenda != []:
        curr_problem = agenda[0]
        agenda.pop(0)
        extensions += 1
        if check_all_constraints(curr_problem) == True and has_empty_domains(curr_problem) == False:
            next_unassigned_var = curr_problem.pop_next_unassigned_var()
            if next_unassigned_var == None:
                return curr_problem.assignments, extensions
            list_of_extensions = create_extensions_with_domain_reduction(curr_problem, next_unassigned_var)
            agenda = list_of_extensions + agenda
    if agenda == []:
        return None, extensions

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

pokemon_problem = get_pokemon_problem()
solution, extensions = solve_constraint_propagate_reduced_domains(pokemon_problem)
ANSWER_4 = extensions


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    dequed_list = []
    csp_copy = csp.copy()
    if queue == None:
        queue = csp_copy.get_all_variables()
    while queue != []:
        var = queue.pop(0)
        dequed_list.append(var)
        reduced_list = eliminate_from_neighbors(csp, var)
        if reduced_list == None:
            return None
        elif reduced_list == []:
            pass
        elif reduced_list != [] and reduced_list != None:
            for neighbor in reduced_list:
                if neighbor not in queue and enqueue_condition_fn(csp, neighbor):
                    queue.append(neighbor)
            if has_empty_domains(csp) == True:
                return None 
    if queue == []:
        return dequed_list

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    if len(csp.domains[var]) == 1:
        return True
    else:
        return False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def create_extensions_with_propagate(enqueue_condition_fn, problem, next_unassigned_var):
    list_of_extensions = []
    list_of_values = problem.get_domain(next_unassigned_var)
    if list_of_values != []:
        for value in list_of_values:
            new_problem = problem.copy()
            new_problem = new_problem.set_assignment(next_unassigned_var, value)####For convenience, also modifies the variable's domain to contain only the assigned value.
            queue = [next_unassigned_var]
            res = propagate(enqueue_condition_fn, new_problem, queue)
            list_of_extensions.append(new_problem)
    return list_of_extensions


def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    if enqueue_condition == None:
        solution, extensions = solve_constraint_dfs(problem)
        return solution, extensions
    else:
        agenda = []
        extensions = 0
        agenda.append(problem)
        while agenda != []:
            curr_problem = agenda[0]
            agenda.pop(0)
            extensions += 1
            if check_all_constraints(curr_problem) == True and has_empty_domains(curr_problem) == False:
                next_unassigned_var = curr_problem.pop_next_unassigned_var()
                if next_unassigned_var == None:
                    return curr_problem.assignments, extensions
                else:
                    list_of_extensions = create_extensions_with_propagate(enqueue_condition, curr_problem, next_unassigned_var)
                    agenda = list_of_extensions + agenda
        if agenda == []:
            return None, extensions



# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

pokemon_problem = get_pokemon_problem()
soluton, extensions = solve_constraint_generic(pokemon_problem, enqueue_condition = condition_singleton)
ANSWER_5 = extensions


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) == 1:
        return True 
    else:
        return False


def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n)!=1:
        return True
    else:
        return False


def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    list_of_constraints = []
    possible_pairs_made = []
    for var1 in variables:
        for var2 in variables:
            if var1!=var2:
                if [var1, var2] not in possible_pairs_made and [var2, var1] not in possible_pairs_made:
                    new_constraint = Constraint(var1, var2, constraint_different)
                    possible_pairs_made.append([var1, var2])
                    possible_pairs_made.append([var2, var1])
                    list_of_constraints.append(new_constraint)
    return list_of_constraints


#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 20
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
