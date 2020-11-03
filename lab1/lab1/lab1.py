# MIT 6.034 Lab 1: Rule-Based Systems
# Written by 6.034 staff

from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain
from data import *
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

#### Part 1: Multiple Choice #########################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '2'

ANSWER_4 = '0'

ANSWER_5 = '3'

ANSWER_6 = '1'

ANSWER_7 = '0'

#### Part 2: Transitive Rule #########################################

# Fill this in with your rule 
transitive_rule = IF( AND('(?x) beats (?y)', '(?y) beats (?z)'), THEN('(?x) beats (?z)'))

# You can test your rule by uncommenting these pretty print statements
#  and observing the results printed to your screen after executing lab1.py
# pprint(forward_chain([transitive_rule], abc_data))
# pprint(forward_chain([transitive_rule], poker_data))
# pprint(forward_chain([transitive_rule], minecraft_data))


#### Part 3: Family Relations #########################################

# Define your rules here. We've given you an example rule whose lead you can follow:
same_rule = IF( AND('person (?x)', 'person (?x)'), THEN ('same (?x) (?x)'))

friend_rule = IF( AND("person (?x)", "person (?y)"), THEN ("friend (?x) (?y)", "friend (?y) (?x)") )

child_rule = IF( AND('parent (?x) (?y)'), THEN ('child (?y) (?x)'))

sibling_rule = IF ( AND('parent (?x) (?y)', 'parent (?x) (?z)', NOT('same (?y) (?z)')),THEN('sibling (?y) (?z)', 'sibling (?z) (?y)'))

cousin_rule = IF( AND('parent (?u) (?x)', 'parent (?v) (?y)', OR('sibling (?u) (?v)', 'sibling (?v) (?u)'), NOT('sibling (?x) (?y)'), NOT('sibling (?y) (?x)')), THEN('cousin (?x) (?y)', 'cousin (?y) (?x)'))

grandparent_rule = IF(AND('parent (?u) (?y)', 'parent (?x) (?u)'), THEN('grandparent (?x) (?y)')) 

grandchild_rule = IF(AND('grandparent (?y) (?x)'), THEN('grandchild (?x) (?y)'))

greatgrandparent_rule = IF(OR(AND('grandparent (?x) (?y)', 'parent (?y) (?z)'), AND('parent (?x) (?y)', 'grandparent (?y) (?z)')), THEN('greatgrandparent (?x) (?z)')) 

nibling_rule = IF(AND('sibling (?x) (?y)', 'child (?z) (?x)'), THEN('nibling (?z) (?y)'))

# Add your rules to this list:
family_rules = [same_rule, friend_rule, child_rule, sibling_rule, cousin_rule, grandparent_rule, grandchild_rule, greatgrandparent_rule, nibling_rule ]

# Uncomment this to test your data on the Simpsons family:
# pprint(forward_chain(family_rules, simpsons_data, verbose=False))

# These smaller datasets might be helpful for debugging:
# pprint(forward_chain(family_rules, sibling_test_data, verbose=True))
# pprint(forward_chain(family_rules, grandparent_test_data, verbose=True))

# The following should generate 14 cousin relationships, representing 7 pairs
# of people who are cousins:
black_family_cousins = [
    relation for relation in
    forward_chain(family_rules, black_data, verbose=False)
    if "cousin" in relation ]

# To see if you found them all, uncomment this line:
# pprint(black_family_cousins)


#### Part 4: Backward Chaining #########################################

# Import additional methods for backchaining
from production import PASS, FAIL, match, populate, simplify, variables

# def ret_hypothesis(rules, hypothesis):
# 	hypothesis = hypothesis
# 	rules = rules
# 	store = OR()
# 	store.append(hypothesis)
# 	for rule in rules:
#    		antecedence = rule.antecedent()
#    		consequence = rule.consequent()
#    		if match(consequence, hypothesis) == {}:
#    			fin_antecedent = antecedence
#    			if isinstance(fin_antecedent, (AND, OR)):
#    				for clause in fin_antecedent:
#    					store.append(ret_hypothesis(rules, fin_antecedent))
#    			else:
#    				store.append(ret_hypothesis, fin_antecedent)
#    		elif match(consequence, hypothesis) == None:
#    			pass
#    		else:
#    			store_binding = match(consequence, hypothesis)
#    			fin_antecedent = populate(antecedence, store_binding)
#    			if isinstance(fin_antecedent, (AND, OR)):
#    				for clause in fin_antecedent:
#    					store.append(ret_hypothesis(rules, fin_antecedent))
#    			else:
#    				store.append(ret_hypothesis(rules, fin_antecedent))
#    	simplify(store)
#    	return store

def ret_hypothesis(rules, hypothesis):
	hypothesis = hypothesis
	rules = rules 
	store = OR()
	store.append(hypothesis)
	for rule in rules:
		antecedence = rule.antecedent()
		consequence = rule.consequent()
		if match(consequence, hypothesis) == {}:
			fin_antecedent = antecedence
			if isinstance(fin_antecedent, AND):
				temp = []
				for clause in fin_antecedent:
					temp.append(ret_hypothesis(rules, clause))
				store.append(AND(temp))

			elif isinstance(fin_antecedent, OR):
				temp = []
				for clause in fin_antecedent:
					temp.append(ret_hypothesis(rules, clause))
				store.append(OR(temp))
			else:
				store.append(ret_hypothesis(rules, fin_antecedent))
		elif match(consequence, hypothesis) == None:
			pass
		else:
			store_binding = match(consequence, hypothesis)
			fin_antecedent = populate(antecedence, store_binding)
			if isinstance(fin_antecedent, AND):
				temp = []
				for clause in fin_antecedent:
					temp.append(ret_hypothesis(rules, clause))
				store.append(AND(temp))
			elif isinstance(fin_antecedent, OR):
				temp = []
				for clause in fin_antecedent:
					temp.append(ret_hypothesis(rules, clause))
					store.append(OR(temp))
			else:
				store.append(ret_hypothesis(rules, fin_antecedent))
	return store






def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """
    hypothesis = hypothesis 
    rules = rules 
    store = ret_hypothesis(rules, hypothesis)
    return simplify(store)
   
# Uncomment this to test out your backward chainer:
# print(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))


#### Survey #########################################

NAME = 'Aneek Das'
COLLABORATORS = ''	
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the tester. DO NOT CHANGE!
transitive_rule_poker = forward_chain([transitive_rule], poker_data)
transitive_rule_abc = forward_chain([transitive_rule], abc_data)
transitive_rule_minecraft = forward_chain([transitive_rule], minecraft_data)
family_rules_simpsons = forward_chain(family_rules, simpsons_data)
family_rules_black = forward_chain(family_rules, black_data)
family_rules_sibling = forward_chain(family_rules, sibling_test_data)
family_rules_grandparent = forward_chain(family_rules, grandparent_test_data)
family_rules_anonymous_family = forward_chain(family_rules, anonymous_family_test_data)
