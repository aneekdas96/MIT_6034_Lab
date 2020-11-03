# MIT 6.034 Lab 9: Boosting (Adaboost)

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    dict_tp = {}
    no_of_points = len(training_points)
    ini_w = make_fraction(1, no_of_points)
    for point in training_points:
    	dict_tp[point] = ini_w
    return dict_tp

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    error_rate_dict = {}
    for classifier in classifier_to_misclassified:
    	misclassified_points = classifier_to_misclassified[classifier]
    	error_rate = 0
    	for point in misclassified_points:
    		weight_of_point = point_to_weight[point]
    		error_rate += weight_of_point
    	error_rate_dict[classifier] = error_rate
    return error_rate_dict

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    # if use_smallest_error == True:
    # 	best_classifier = min(classifier_to_error_rate, key = lambda point:make_fraction(classifier_to_error_rate[point]))
    # else:
    # 	min_best_classifier = min(classifier_to_error_rate, key = lambda point:(make_fraction(classifier_to_error_rate[point]), point))
    # 	max_best_classifier = max(classifier_to_error_rate, key = lambda point:(make_fraction(classifier_to_error_rate[point]), point))
    # 	dist_min_classifier = abs(make_fraction(1, 2) - make_fraction(classifier_to_error_rate[min_best_classifier]))
    # 	dist_max_classifier = abs(make_fraction(1, 2) - make_fraction(classifier_to_error_rate[max_best_classifier]))
    # 	if make_fraction(dist_min_classifier) > make_fraction(dist_max_classifier):
    # 		best_classifier = min_best_classifier
    # 	elif make_fraction(dist_max_classifier) > make_fraction(dist_min_classifier):
    # 		best_classifier = max_best_classifier
    # 	else:
    # 		best_classifier =  min(min_best_classifier, max_best_classifier)
    # if make_fraction(classifier_to_error_rate[best_classifier]) == make_fraction(1, 2) or best_classifier =='':
    # 	# print('error rate 1/2, no good classifier')
    # 	raise NoGoodClassifiersError
    # else:
    # 	return best_classifier
    classifier_to_error_rate_new = {}
    list1 = sorted(classifier_to_error_rate.keys())
    for key in list1:
        classifier_to_error_rate_new[key] = classifier_to_error_rate[key]
    classifier_min_error = min(classifier_to_error_rate_new.items(), key=lambda x:x[1])
    best = ""
    if use_smallest_error == True:
        best = classifier_min_error[0]
    else:
        furthest = 0
        for key in classifier_to_error_rate_new.keys():
            if abs(classifier_to_error_rate_new[key]-Fraction(1,2)) > furthest:
                furthest = abs(classifier_to_error_rate_new[key]-Fraction(1,2))
                best = key
    if best == ""  or classifier_to_error_rate_new[best] == Fraction(1,2):
        raise NoGoodClassifiersError
    else:
        return best
    

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 1:
    	return -INF
    elif error_rate == 0:
    	return INF
    else:
    	store = 0.5 * ln((1-error_rate)/error_rate)
    	return store

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified_points = set()
    all_classifiers = []
    dict_c = {}
    all_vps = []
    for item in H:
    	all_classifiers.append(item[0])
    	all_vps.append(item[1])
    	dict_c[item[0]] = item[1]
    for point in training_points:
    	correct_classifiers = []
    	wrong_classifiers = []
    	correct_score = 0
    	wrong_score = 0
    	for classifier in all_classifiers:
    		classifier_errors = classifier_to_misclassified[classifier]
    		if point in classifier_errors:
    			wrong_classifiers.append(classifier)
    		else:
    			correct_classifiers.append(classifier)
    	for c1 in wrong_classifiers:
    		wrong_score += dict_c[c1]
    	for c2 in correct_classifiers:
    		correct_score += dict_c[c2]
    	if wrong_score >= correct_score:
    		misclassified_points.add(point)
    return misclassified_points

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassifications = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    no_misclassified = len(misclassifications)
    if no_misclassified > mistake_tolerance:
    	return False
    else:
    	return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    all_points = list(point_to_weight.keys())
    new_weights_dict = {}
    for point in all_points:
    	old_weight = point_to_weight[point]
    	try:
    		if point not in misclassified_points:
    			new_weight = make_fraction(1, 2) * make_fraction(1, (1-error_rate)) * old_weight
    		else:
    			new_weight = make_fraction(1, 2) * make_fraction(1, error_rate) * old_weight
    	except Exception as e:
    		new_weight = INF
    	new_weights_dict[point] = new_weight
    return new_weights_dict
	
#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    ensemble_classifier = []
    training_points_to_weights = initialize_weights(training_points)
    rounds = 0
    while rounds < max_rounds:
    	classifier_to_error_rate = calculate_error_rates(training_points_to_weights, classifier_to_misclassified)
    	try:
	    	best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
	    	error_rate = classifier_to_error_rate[best_classifier]
	    	if error_rate == make_fraction(1, 2):
	    		return ensemble_classifier
	    	voting_power = calculate_voting_power(error_rate)
	    	ensemble_classifier.append((best_classifier, voting_power))
	    	misclassified_points = classifier_to_misclassified[best_classifier]
	    	training_points_to_weights = update_weights(training_points_to_weights, misclassified_points, error_rate)
    	except Exception as e:
    		return ensemble_classifier 
    	if is_good_enough(ensemble_classifier, training_points, classifier_to_misclassified, mistake_tolerance):
    		return ensemble_classifier
    	else:
    		rounds += 1
    return ensemble_classifier
    # point_to_weight = initialize_weights(training_points)
    # classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
    # best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
    # error_rate = classifier_to_error_rate[best_classifier]
    # vp = calculate_voting_power(error_rate)
    # H = [(best_classifier, vp)]
    # count = 1
    # while is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance) == False and count < max_rounds:
    #     try:
    #         point_to_weight = update_weights(point_to_weight, classifier_to_misclassified[best_classifier], error_rate)
    #         classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
    #         best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
    #         error_rate = classifier_to_error_rate[best_classifier]
    #         vp = calculate_voting_power(error_rate)
    #         H.append((best_classifier, vp))
    #     except:
    #         return H
    #     count = count + 1
    # return H    

#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 4
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
