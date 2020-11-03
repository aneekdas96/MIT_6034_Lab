# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    if id_tree.is_leaf():
        classification = id_tree.get_node_classification()
        return classification
    else:
        child_node = id_tree.apply_classifier(point)
        return id_tree_classify_point(point, child_node)


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    feature_values = []
    for point in data:
        classification = classifier.classify(point)
        if classification not in feature_values:
            feature_values.append(classification)
    classification_dict = {}
    for feature_value in feature_values:
        classification_dict[feature_value] = []
    for point in data:
        classification = classifier.classify(point)
        classification_dict[classification].append(point)
    return classification_dict

#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    number_of_data_points = len(data)
    classification_dict = split_on_classifier(data, target_classifier)
    disorder = 0
    for classification in classification_dict:
        number_of_classification = len(classification_dict[classification])
        disorder += (-(number_of_classification/number_of_data_points)*log2(number_of_classification/number_of_data_points))
    return disorder


def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    try:
        total_weighted_disorder = 0
        number_of_data_points = len(data)
        branches = split_on_classifier(data, test_classifier)
        for branch in branches:
            number_of_items_in_branch = len(branches[branch])
            weight_factor = number_of_items_in_branch/number_of_data_points
            list_of_samples_in_branch = branches[branch]
            disorder_of_branch = branch_disorder(list_of_samples_in_branch, target_classifier)
            total_weighted_disorder += (weight_factor * disorder_of_branch)
        return total_weighted_disorder
    except Exception as e:
        return None

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    best_disorder_score = 10000000
    best_classifier = None
    try:
        for classifier in possible_classifiers:
            total_disorder = average_test_disorder(data, classifier, target_classifier)
            if total_disorder < best_disorder_score:
                best_classifier = classifier
                best_disorder_score = total_disorder
            else:
                pass
        if best_classifier!=None:
            branches = split_on_classifier(data, best_classifier)
            if len(branches) == 1:
                raise NoGoodClassifiersError
            else:
                return best_classifier
    except Exception as e:
        raise NoGoodClassifiersError

## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    classification_dict = split_on_classifier(data, target_classifier)

    if len(classification_dict.keys()) == 1:
        classification = target_classifier.classify(data[0])
        id_tree_node.set_node_classification(classification)
    else:
        try:
            best_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
            split_data = split_on_classifier(data, best_classifier)
            id_tree_node = id_tree_node.set_classifier_and_expand(best_classifier, split_data)
            branches_for_node = id_tree_node.get_branches()
            for branch in branches_for_node:
                construct_greedy_id_tree(split_data[branch], possible_classifiers, target_classifier, branches_for_node[branch])
        except NoGoodClassifiersError:
            return id_tree_node
    return id_tree_node


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    sum_of_products = 0
    if u!= None:
        if v!= None:
            for combo in zip(u, v):
                sum_of_products += (combo[0] * combo[1])
    return sum_of_products


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    dot_product_v = dot_product(v, v)
    return math.sqrt(dot_product_v)

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    sum_distance = 0
    for combo in zip(point1, point2):
        sum_distance += ((combo[0] - combo[1])**2)
    return math.sqrt(sum_distance)


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    sum_distance = 0
    for combo in zip(point1, point2):
        sum_distance += abs(combo[0] - combo[1])
    return sum_distance

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    different_components = 0 
    for combo in zip(point1, point2):
        if combo[0] != combo[1]:
            different_components += 1
        else:
            pass
    return different_components

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    cos_dist = 0
    length_point1 = norm(point1)
    length_point2 = norm(point2)
    cos_dist = 1 - (dot_product(point1, point2)/(length_point1 * length_point2))
    return cos_dist



#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    points_and_scores = []
    k_closest_points = []
    for item in data:
        item_score = distance_metric(point, item)
        points_and_scores.append([item, item_score])
    points_and_scores = sorted(points_and_scores, key = lambda item:(item[1], item[0].coords))
    for i in range(k):
        k_closest_points.append(points_and_scores[i][0])
    return k_closest_points

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    k_closest_points = get_k_closest_points(point, data, k, distance_metric)
    classification_counts = {}
    for item in k_closest_points:
        classification_type = item.classification
        if classification_type not in classification_counts:
            classification_counts[classification_type] = 0
        else:
            classification_counts[classification_type] += 1
    classification_counts = sorted(classification_counts, key = classification_counts.get)
    return classification_counts[-1]

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    fraction_correct = 0.00
    correctly_classified = 0
    for i, test_data in enumerate(data):
        training_data = []
        for j in range(len(data)):
            if j!=i:
                training_data.append(data[j])
        observed_classification = knn_classify_point(test_data, training_data, k, distance_metric)
        actual_classification = test_data.classification
        if observed_classification == actual_classification:
            correctly_classified += 1
    fraction_correct = float(correctly_classified/len(data))
    return fraction_correct

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    metrics_and_scores = []
    possible_metrics = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]
    for k in range(1, len(data)):
        for metric in possible_metrics:
            cross_validation_score = cross_validate(data, k, metric)
            metrics_and_scores.append([k, metric, cross_validation_score])
    sorted_metrics = sorted(metrics_and_scores, key = lambda item:item[2])
    return (sorted_metrics[-1][0], sorted_metrics[-1][1])


## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = 'Overfitting'
kNN_ANSWER_2 = 'Underfitting'
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 8
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
