# MIT 6.034 Lab 8: Bayesian Inference

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = set()
    list_of_vars = [var]
    while list_of_vars != []:
    	current_var = list_of_vars[0]
    	list_of_vars.pop(0)
    	parents_of_current_var = net.get_parents(current_var)
    	for parent in parents_of_current_var:
    		ancestors.add(parent)
    		list_of_vars.append(parent)
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = set()
    list_of_vars = [var]
    while list_of_vars != []:
    	current_var = list_of_vars[0]
    	list_of_vars.pop(0)
    	children_of_current_var = net.get_children(current_var)
    	for child in children_of_current_var:
    		descendants.add(child)
    		list_of_vars.append(child)
    return descendants 	

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    children = get_descendants(net, var)
    variables = set(net.get_variables())
    non_descendants = variables - children 
    non_descendants.remove(var)
    return non_descendants


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    #condition1 : all parents included and no descendants
    givens_keys = set(givens.keys())
    new_givens = {}
    flag = False
    non_descendants = get_nondescendants(net, var)
    parents = net.get_parents(var)
    descendants = get_descendants(net, var)
    for item in givens_keys:
    	if item in descendants:
    		flag = True
    if parents - givens_keys == set() and givens_keys - descendants == givens_keys:
    	to_remove = non_descendants - parents
    	for item in givens_keys:
    		if item not in to_remove:
    			new_givens[item] = givens[item]
    	return new_givens
    elif parents - givens_keys != set() or flag == True:
    	return givens
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    try:
    	simplified_givens = None
    	if givens != None:
    		var = list(hypothesis.keys())[0]
    		simplified_givens = simplify_givens(net, var, givens)
    	prob = net.get_probability(hypothesis, simplified_givens)
    	return prob
    except ValueError:
    	raise LookupError
 
def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    list_of_variables = net.topological_sort()
    total_prob = 1
    for index, var in enumerate(list_of_variables):
    	hyp_temp = {}
    	given_var = {}
    	hyp_temp[var] = hypothesis[var]
    	given_parents = net.get_parents(var)
    	for parent in given_parents:
    		given_var[parent] = hypothesis[parent]
    	total_prob = total_prob * probability_lookup(net, hyp_temp, given_var)
    return total_prob

def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    sum_prob = 0
    vars_in_net = net.get_variables()
    vars_in_hyp = list(hypothesis.keys())
    other_vars = [] 
    for var in vars_in_net:
    	if var not in vars_in_hyp:
    		other_vars.append(var)
    combine_dicts = net.combinations(other_vars, hypothesis)
    for dict_var in combine_dicts:
    	sum_prob = sum_prob + probability_joint(net, dict_var)
    return sum_prob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    #base case, no conditionals
    if givens == None:
    	return probability_marginal(net, hypothesis)
    hyp_vars = list(hypothesis.keys())
    given_vars = list(givens.keys())
    common_vars = [var for var in hyp_vars if var in given_vars]
    #edge case, one variable common in hypothesis and givens
    if givens!=None:
	    if len(hyp_vars) == 1 and len(given_vars) == 1 and hyp_vars[0] == given_vars[0]:
	    	if hypothesis[hyp_vars[0]] == givens[given_vars[0]]:
	    		return 1
	    	else:
	    		return 0
    #eliminating common vars
    for var in common_vars:
    	if hypothesis[var] == givens[var]:
    		hypothesis.pop(var)
    	elif hypothesis[var] != givens[var]:
    		return 0
    try:
    	prob = probability_lookup(net, hypothesis, givens)
    	return prob
    except Exception as e:
    	new_hyp = dict(hypothesis, **givens)
    	num = probability_marginal(net, new_hyp)
    	den = probability_marginal(net, givens)
    	div_val = num/den
    	return div_val

def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    hyp_vars = set(list(hypothesis.keys()))
    all_vars = set(net.get_variables())
    if all_vars - hyp_vars== set():
    	return probability_joint(net, hypothesis)
    elif hyp_vars - all_vars != set() and given == None:
    	return probability_marginal(net, hypothesis)
    else:
    	return probability_conditional(net, hypothesis, givens)

#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    all_vars = net.topological_sort()
    net_params = []
    for var in all_vars:
    	#getting number of columns
    	no_values = len(net.get_domain(var))
    	if no_values == 1:
    		columns = 1 
    	elif no_values >1:
    		columns = no_values - 1
    	#getting number of rows
    	parents = net.get_parents(var)
    	no_parents = len(parents)
    	if parents == set():
    		rows = 1
    	else:
    		rows = 1
    		for parent in parents:
    			parents_values = len(net.get_domain(parent))
    			rows = rows * parents_values
    	params = rows * columns
    	net_params.append(params)
    store = sum(net_params)
    return store

#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    #marginal case
    flag = True
    if givens == None:
        values_var1 = net.get_domain(var1)
        values_var2 = net.get_domain(var2)
        for val1 in values_var1:
            for val2 in values_var2:
                hypothesis = {var1:val1, var2:val2}              
                actual_marginal = probability_marginal(net, hypothesis)
                var1_marginal = probability_marginal(net, {var1: val1})
                var2_marginal = probability_marginal(net, {var2: val2})
                prod_marginal = var1_marginal * var2_marginal
                if approx_equal(actual_marginal, prod_marginal, epsilon=0.0000000001) == False:
                	flag = False
        if flag == False:
        	return False
        else:
        	return True

    else:#conditional case
        values_var1 = net.get_domain(var1)
        values_var2 = net.get_domain(var2)
        count1 = 0
        for val1 in values_var1:
            for val2 in values_var2:
                hypothesis = {var1:val1}
                new_givens = givens.copy()
                new_givens[var2] = val2
                eval_val = probability_conditional(net, hypothesis, givens)
                orig_val = probability_conditional(net, hypothesis, new_givens)
                if approx_equal(eval_val, orig_val, epsilon=0.0000000001) == False:
                	flag = False
        if flag == False:
        	return False
        else:
        	return True
            
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    if givens != None:
        list_of_val = [var1, var2]+list(givens.keys())
    
    else:
        list_of_val = [var1, var2]
    
    for val in list_of_val:
       ancestors = get_ancestors(net, val)
       list_of_val += list(ancestors)
    list_of_val = set(list_of_val)
    sub_net = net.subnet(list_of_val)
    parents_to_marry = []
    
    for v1 in sub_net.get_variables():
        c1 = sub_net.get_children(v1)
        for v2 in sub_net.get_variables():
            c2 = sub_net.get_children(v2)
            if v1!=v2 and len(c1.intersection(c2)) != 0:
                parents_to_marry.append((v1, v2))
    
    for pair in parents_to_marry:
        sub_net.link(pair[0],pair[1]) 
    sub_net.make_bidirectional()

    if givens != None:
        for g in givens:
            sub_net.remove_variable(g)
    if sub_net.find_path(var1, var2):
        return False
    else:
        return True



#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
