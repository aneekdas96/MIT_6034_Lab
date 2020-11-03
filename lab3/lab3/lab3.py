# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    game_over = 0
    if board.count_pieces() == 42:
        gover = 1
        return True
    chains = board.get_all_chains()
    for chain in chains:
        if len(chain) >= 4:
            game_over = 1
            return True
    if game_over == 0:
        return False

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    list_of_boards = []
    if is_game_over_connectfour(board):
        list_of_boards = []
        return list_of_boards

    for i in range(board.num_cols):
        if board.is_column_full(i):
            pass
        else:
            nxt_board = board.add_piece(i)
            list_of_boards.append(nxt_board)
    return list_of_boards

def tie_condition_connectfour(board):
    chain_over_four = 0
    chains = board.get_all_chains()
    for chain in chains:
        if len(chain) >= 4:
            chain_over_four = 1
    if board.count_pieces() >= 42 and chain_over_four == 0:
        return True
    else:
        return False

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if is_game_over_connectfour(board) == True:
        if tie_condition_connectfour(board) == True:
            return 0
        elif is_current_player_maximizer == True:
            return -1000
        elif is_current_player_maximizer == False:
            return +1000
    else:
        return None

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    if is_game_over_connectfour(board) == True:
        if tie_condition_connectfour(board) == True:
            return 0
        elif is_current_player_maximizer == False:
            score = 1000 + (42 - board.count_pieces())
            return score
        elif is_current_player_maximizer == True:
            score = -1000 - (42 - board.count_pieces())  
            return score
    else:
        return None

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    score_for_player = 0
    score_for_opponent = 0
    chains = board.get_all_chains()
    chains_for_player = []
    chains_for_opponent = []
    for chain in chains:
        if 1 in chain:
            chains_for_player.append(chain)
        elif 2 in chain:
            chains_for_opponent.append(chain)
    for chain in chains_for_player:
        if len(chain) == 1:
            score_for_player += 1
        elif len(chain) == 2:
            score_for_player += 5
        elif len(chain) >=3:
            score_for_player += 10
    for chain in chains_for_opponent:
        if len(chain) == 1:
            score_for_opponent += 1
        elif len(chain) == 2:
            score_for_opponent += 5
        elif len(chain) >= 3:
            score_for_opponent += 10
    heuristic_score = score_for_player - score_for_opponent
    if is_current_player_maximizer == True:
        return heuristic_score
    else:
        return -heuristic_score

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def create_new_path(path, nbor):
    temp = []
    for node in path:
        temp.append(node)
    temp.append(nbor)
    return temp

def join_node_with_path(parent, path):
    temp = []
    temp.append(parent)
    for node in path:
        temp.append(node)
    return temp

def extensions(path):
    state = path[-1]
    list_of_children = state.generate_next_states()
    list_of_paths = []
    for node in list_of_children:
        node_path = create_new_path(path, node)
        list_of_paths.append(node_path)
    return list_of_paths

def ret_heuristic(path, static_eval):
    curr_state = path[-1]
    curr_state_snap = curr_state.get_snapshot()
    static_eval += 1
    heuristic_score = heuristic_connectfour(curr_state_snap, True)
    return heuristic_score

def ret_dfs(agenda, the_best_path, the_best_score, static_eval):
    if agenda != []:
        current_path = agenda[0]
        last_state = current_path[-1]
        if last_state.is_game_over():
            score_of_state = last_state.get_endgame_score(True)
            static_eval += 1
            if score_of_state > the_best_score:
                the_best_score = score_of_state
                the_best_path = current_path
            agenda.pop(0)
            return ret_dfs(agenda, the_best_path, the_best_score, static_eval)
        else:
            list_of_extension = extensions(current_path)                
            agenda.pop(0)
            for i in range(len(list_of_extension)-1, -1, -1):
                agenda.insert(0, list_of_extension[i])
            return ret_dfs(agenda, the_best_path, the_best_score, static_eval)
    elif agenda == []:
        return the_best_path, the_best_score, static_eval


def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    agenda = []
    agenda.append([state])
    the_best_path = []
    the_best_score = 0
    static_eval = 0
    best_path, score, static_eval = ret_dfs(agenda, the_best_path, the_best_score, static_eval)
    return [best_path, score, static_eval]


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def ret_minimax(parent_node, maximize, best_value, best_path, static_eval):
    if parent_node.is_game_over():
        best_value = parent_node.get_endgame_score(maximize)
        static_eval += 1
        best_path = [parent_node]
        return best_value,best_path, static_eval
    list_of_child_paths = extensions([parent_node])
    store = []
    if maximize == True:
        maximize = False
        for child_path in list_of_child_paths:
            current_child = child_path[-1]
            val, path, static_eval = ret_minimax(current_child, maximize, best_value, best_path, static_eval)
            store.append([val, path])
        store = sorted(store, key = lambda item:item[0])
        best_path = store[-1][1]
        best_path_add = join_node_with_path(parent_node, best_path)
        best_value = store[-1][0]
        return best_value, best_path_add, static_eval
    elif maximize == False:
        maximize = True
        for child_path in list_of_child_paths:
            current_child = child_path[-1]
            val, path, static_eval = ret_minimax(current_child, maximize, best_value, best_path, static_eval)
            store.append([val, path])
        store = sorted(store, key = lambda item:item[0])
        best_path = store[0][1]
        best_path_add = join_node_with_path(parent_node, best_path)
        best_value = store[0][0]
        return best_value, best_path_add, static_eval

def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    best_value = 0
    best_path = []
    static_eval = 0
    parent_node = state
    best_value, best_path, static_eval = ret_minimax(parent_node, maximize, best_value, best_path, static_eval)
    return [best_path, best_value, static_eval]


# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))

def ret_minimax_with_heuristic(heuristic_fn, current_depth, depth, parent_node, maximize, best_value, best_path, static_eval):
    if parent_node.is_game_over():
        best_value = parent_node.get_endgame_score(maximize)
        static_eval += 1
        best_path = [parent_node]
        return best_value,best_path, static_eval

    elif current_depth == depth:
        snap_of_state = parent_node.get_snapshot()
        best_value = heuristic_fn(snap_of_state, maximize)
        static_eval += 1
        best_path = [parent_node]
        return best_value, best_path, static_eval

    elif current_depth < depth:
        current_depth += 1
        list_of_child_paths = extensions([parent_node])
        store = []
        if maximize == True:
            maximize = False
            for child_path in list_of_child_paths:
                current_child = child_path[-1]
                val, path, static_eval = ret_minimax_with_heuristic(heuristic_fn, current_depth, depth, current_child, maximize, best_value, best_path, static_eval)
                store.append([val, path])
            store = sorted(store, key = lambda item:item[0])
            best_path = store[-1][1]
            best_path_add = join_node_with_path(parent_node, best_path)
            best_value = store[-1][0]
            return best_value, best_path_add, static_eval
        elif maximize == False:
            maximize = True
            for child_path in list_of_child_paths:
                current_child = child_path[-1]
                val, path, static_eval = ret_minimax_with_heuristic(heuristic_fn, current_depth, depth, current_child, maximize, best_value, best_path, static_eval)
                store.append([val, path])
            store = sorted(store, key = lambda item:item[0])
            best_path = store[0][1]
            best_path_add = join_node_with_path(parent_node, best_path)
            best_value = store[0][0]
            return best_value, best_path_add, static_eval

def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    depth = depth_limit
    current_depth = 0
    best_value = 0
    best_path = []
    static_eval = 0
    parent_node = state
    best_value, best_path, static_eval = ret_minimax_with_heuristic(heuristic_fn, current_depth, depth, parent_node, maximize, best_value, best_path, static_eval)
    return [best_path, best_value, static_eval]


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    raise NotImplementedError


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = 'Aneek Das'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 12
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
