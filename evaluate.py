import numpy as np
import random
from env2048 import Game2048Env, move_board
PERFECT_SNAKE = [[0, 0, 0, 1/2**12],
                [1/2**8, 1/2**9, 1/2**10, 1/2**11],
                [1/2**7, 1/2**6, 1/2**5, 1/2**4],
                [1,1/2,1/2**2,1/2**3]]

def create_env_from_state(state, score):
    new_env = Game2048Env()
    new_env.board = state.copy()
    new_env.score = score
    return new_env
def is_game_over_board(board):
    new_env = create_env_from_state(board, 0)
    return new_env.is_game_over()

def expectimax(board, score, depth, is_player_turn, approximator):
    if depth == 0 or is_game_over_board(board):
        return value_shaping(board)
    
    if is_player_turn:
        best_value = -float("inf")
        for action in range(4):
            new_board, move_score, moved = move_board(board, action)
            if not moved:
                continue
            value = expectimax(new_board, score + move_score, depth - 1, False, approximator)
            best_value = max(best_value, value)
        return best_value if best_value != -float("inf") else value_shaping(board)
    else:
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            return value_shaping(board)
        expected_value = 0
        for cell in empty_cells:
            for tile, prob in [(2, 0.9), (4, 0.1)]:
                new_board = board.copy()
                new_board[cell] = tile
                value = expectimax(new_board, score, depth - 1, True, approximator)
                expected_value += (prob / len(empty_cells)) * value
        return expected_value

def evaluate(env, approximator, depth=3):
    board = env.board.copy()
    best_value = -float("inf")
    best_action = None
    for action in range(4):
        new_board, move_score, moved = move_board(board, action)
        if not moved:
            continue
        value = expectimax(new_board, env.score + move_score, depth - 1, False, approximator)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action

def value_shaping(board):
    shaping_value = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            shaping_value[0] += board[i][j] * PERFECT_SNAKE[i][j]
            shaping_value[1] += board[3-j][i] * PERFECT_SNAKE[i][j]
            shaping_value[2] += board[3-i][3-j] * PERFECT_SNAKE[i][j]
            shaping_value[3] += board[j][3-i] * PERFECT_SNAKE[i][j]

    return np.max(shaping_value)