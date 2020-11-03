# MIT 6.034 Lab 7: Support Vector Machines

from svm_data import *
from functools import reduce
import math

#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    sop = 0
    if u != None:
    	if v != None:
    		for combo in zip(u, v):
    			sop  += (combo[0] * combo[1])
    return sop

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    dot_product_v = dot_product(v, v)
    return math.sqrt(dot_product_v)


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    x = point.coords 
    w = svm.w 
    b = svm.b 
    return (dot_product(w, x) + b)

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    res = positiveness(svm, point)
    if res > 0:
    	return 1
    elif res < 0:
    	return -1
    else:
    	return 0

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    violation_set = set()
    train_points = svm.training_points
    support_vecs = svm.support_vectors
    all_points = train_points + support_vecs
    for point in all_points: 
    	flag = 0
    	class_x = point.classification
    	pos_x = positiveness(svm, point)
    	if point in support_vecs and class_x != pos_x:
    		flag = 1
    	elif point in train_points and -1<pos_x<1:
    		flag = 1
    	if flag == 1:
    		violation_set.add(point)
    return violation_set

#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    violation_set = set()
    train_points = svm.training_points
    support_vecs = svm.support_vectors
    for point in train_points:
    	flag = 0
    	alpha_val = point.alpha
    	if alpha_val < 0:
    		flag = 1
    	elif point in support_vecs and alpha_val <= 0:
    		flag = 1
    	elif point in train_points and point not in support_vecs and alpha_val != 0:
    		flag = 1
    	if flag == 1:
    		violation_set.add(point)
    return violation_set


def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    train_points = svm.training_points
    boundary_constraint = 0
    point_constraint = []
    for i in range(len(train_points[0].coords)):
    	point_constraint.append(0)
    for point in train_points:
    	class_x = point.classification
    	x_coords = point.coords 
    	x_alpha = point.alpha
    	boundary_constraint += (class_x * x_alpha)
    	prod = scalar_mult(class_x * x_alpha, x_coords)
    	point_constraint = vector_add(point_constraint, prod)
    if boundary_constraint == 0 and svm.w == point_constraint:
    	return True
    else:
    	return False

#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    misclassified_set = set()
    train_points = svm.training_points 
    for point in train_points:
    	act_class = point.classification
    	obs_class = classify(svm, point)
    	if act_class != obs_class:
    		misclassified_set.add(point)
    return misclassified_set

#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    
    train_points = svm.training_points
    weight_update = []
    neg_b = []
    pos_b = []
    s_vecs = []
    
    for _ in range(len(train_points[0].coords)):
    	weight_update.append(0)
    
    for point in train_points:
    	x_alpha = point.alpha
    	if x_alpha > 0:
    		s_vecs.append(point)
    svm.support_vectors = s_vecs

    # for point in train_points:
    # 	x_alpha = point.alpha 
    # 	class_x = classify(svm, point)
    # 	x_coords = point.coords
    # 	temp = class_x * x_alpha
    # 	prod = scalar_mult(temp, x_coords)
    # 	weight_update = vector_add(weight_update, prod)
    # 	bias = class_x - dot_product(svm.w, x_coords)
    # 	if point.classification == -1 and point in svm.support_vectors:
    # 		neg_b.append(bias)
    # 	elif point.classification == 1 and point in svm.support_vectors:
    # 		pos_b.append(bias)

    # min_b = min(neg_b)
    # max_b = max(pos_b)
    # bias_update = (max_b + min_b)/2
    # svm.w = weight_update
    # svm.b = bias_update
    # return svm

    for point in train_points:
    	x_alpha = point.alpha 
    	class_x = point.classification
    	x_coords = point.coords
    	prod = scalar_mult(class_x * x_alpha, x_coords)
    	weight_update = vector_add(weight_update, prod)

    svm.w = weight_update

    for point in train_points:
    	x_coords = point.coords
    	class_x = point.classification
    	bias = class_x - dot_product(svm.w, x_coords)
    	if class_x == -1 and point in svm.support_vectors:
    		neg_b.append(bias)
    	elif class_x == 1 and point in svm.support_vectors:
    		pos_b.append(bias)

    min_b = min(neg_b)
    max_b = max(pos_b)
    bias_update = (max_b + min_b)/2
    svm.b = bias_update
    return svm


#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 1.5
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
