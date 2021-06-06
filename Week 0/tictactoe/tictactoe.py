"""
Tic Tac Toe Player
"""

import math
import copy
import numpy as np

X = "X"
O = "O"

EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board."""
    num_x = 0
    num_o = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                num_x = num_x + 1
            if board[i][j] == O:
                num_o = num_o + 1

    if num_x == num_o:
        return X

    elif num_x > num_o:
        return O


def actions(board):

    """
    Returns set of all possible actions (i, j) available on the board.
    """

    actions_list = []
    if terminal(board):
        return None

    else:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != X and board[i][j] != O:
                    actions_list.append((i, j))

    return actions_list


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    result function takes a board and an action as input, and should return a new board state, without modifying the
    original board.
    """

    new_board = copy.deepcopy(board)

    if action not in actions(board):
        raise Exception("Invalid Action")

    else:
        new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    If there is no winner of the game (either because the game is in progress, or because it ended in a tie),
    the function should return None.

    check for 3 consecutive rows, columns....>
    """

    # transposition to check rows, then columns
    for newBoard in [board, np.transpose(board)]:
        check_winner = check_row(newBoard)

        if check_winner in {O, X}:
            return check_winner

    return check_diagonals(board)


def terminal(board):
    """
    board as input, and Returns True if game is over, False otherwise.
    If the game is over, either because someone has won the game or because all cells have been filled without anyone
    winning, the function should return True
    """

    if (winner(board) in {X, O}) or (not is_any_cell_empty(board)):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.

    """
    game_winner = winner(board)

    if game_winner == 'X':
        return 1

    elif game_winner == 'O':
        return -1

    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    The move returned should be the optimal action (i, j) that is one of the allowable actions on the board.
    If multiple moves are equally optimal, any of those moves is acceptable.
    If the board is a terminal board, the minimax function should return None.

    get the current player: player(board)
    if X, -> pick action max of (Min-Value(Result(s, a)).
    if O ->  pick action lowest of Max-Value(Result(s, a)).

    """
    best_move = None

    current_turn = player(board)
    alpha = -math.inf
    beta = math.inf

    if terminal(board):
        return None

    # Maximizing Player
    if current_turn == X:
        max_weight = -math.inf

        for action in actions(board):
            weight = get_min_value((result(board, action)), alpha, beta)

            if weight > max_weight:
                max_weight = weight
                best_move = action


    # Minimizing Player
    elif current_turn == O:
        min_weight = math.inf
        for action in actions(board):
            weight = get_max_value((result(board, action)), alpha, beta)

            if weight < min_weight:
                min_weight = weight
                best_move = action

    else:
        return  # Not a possible condition

    return best_move


def get_max_value(board, alpha, beta):
    """Helper function to get the value of the move with the maximum weight """
    max_value = -math.inf
    max_move = None

    action_weights = []
    if terminal(board):
        return utility(board)

    for action in actions(board):
        max_value = max(max_value, get_min_value(result(board, action), alpha, beta))
        alpha = max_value

        if beta <= alpha:
            break

    return max_value


def get_min_value(board, alpha, beta):
    """ Helper function to get the value of the move with the minimum weight """
    min_value = math.inf
    min_move = None

    action_weights = []
    if terminal(board):
        return utility(board)

    for action in actions(board):
            min_value = min(min_value, get_max_value(result(board, action), alpha, beta))
            beta = min_value

            if beta <= alpha:
                break

    return min_value


def is_any_cell_empty(board):
    """ returns true if cells are all occupied(no cell has the value EMPTY)"""
    return any(EMPTY in sublist for sublist in board)


def check_row(board):
    """ checking if all the elements in a row are the same, 0 if no winner"""
    for row in board:
        if len(set(row)) == 1:
            return row[0]

    return -1


def check_diagonals(board):
    """ checking if all the elements in the diagonal are the same, None if no elements are the same """
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]

    if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
        return board[0][len(board)-1]

    return None

