import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from functools import lru_cache
import numba as nb

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}
@nb.jit(nopython=True)
def move_board(board, direction):
    """
    direction: 0: 上, 1: 下, 2: 左, 3: 右
    """
    size = board.shape[0]
    new_board = np.copy(board)
    total_score = 0

    if direction == 2:  # 左
        for i in range(size):
            row, score = process_row_cache(board[i])
            new_board[i] = row
            total_score += score
    elif direction == 3:  # 右
        for i in range(size):
            reversed_row = board[i, ::-1]
            row, score = process_row_cache(reversed_row)
            new_board[i] = row[::-1]
            total_score += score
    elif direction == 0:  # 上
        for j in range(size):
            col, score = process_row_cache(board[:, j])
            new_board[:, j] = col
            total_score += score
    elif direction == 1:  # 下
        for j in range(size):
            reversed_col = board[::-1, j]
            col, score = process_row_cache(reversed_col)
            new_board[:, j] = col[::-1]
            total_score += score

    moved = not np.array_equal(board, new_board)
    return new_board, total_score, moved


@nb.jit(nopython=True)
def process_row_cache(row_tuple):
    row = list(row_tuple)
    non_zero = [x for x in row if x != 0]
    merged = []
    score_gained = 0
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
            merged_val = non_zero[i] * 2
            merged.append(merged_val)
            score_gained += merged_val
            skip = True
        else:
            merged.append(non_zero[i])
    new_row = merged + [0] * (len(row) - len(merged))
    return new_row, score_gained

def process_np_row(row):
    new_row_tuple, score_gained = process_row_cache(tuple(row))
    # new_row_tuple = tuple(new_row_tuple)
    new_row = np.array(new_row_tuple)
    return new_row, score_gained

@nb.jit(nopython=True)
def is_move_possible(board, direction):
    size = board.shape[0]
    
    if direction == 2:  # 左
        for i in range(size):
            original_row = board[i].copy()
            processed_row, _ = process_row_cache(original_row)
            if not np.array_equal(original_row, processed_row):
                return True
        return False
    
    elif direction == 3:  # 右
        for i in range(size):
            original_row = board[i].copy()
            reversed_row = original_row[::-1]
            processed_row, _ = process_row_cache(reversed_row)
            if not np.array_equal(original_row, processed_row[::-1]):
                return True
        return False
    
    elif direction == 0:  # 上
        for j in range(size):
            original_col = board[:, j].copy()
            processed_col, _ = process_row_cache(original_col)
            if not np.array_equal(original_col, processed_col):
                return True
        return False
    
    elif direction == 1:  # 下
        for j in range(size):
            original_col = board[:, j].copy()
            reversed_col = original_col[::-1]
            processed_col, _ = process_row_cache(reversed_col)
            if not np.array_equal(original_col, processed_col[::-1]):
                return True
        return False


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # 行動空間: 0: 上, 1: 下, 2: 左, 3: 右
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True


    def move(self, action):
        assert self.action_space.contains(action), "Invalid action"
        new_board, score_gained, moved = move_board(self.board, action)
        if moved:
            self.board = new_board
            self.score += score_gained
        return self.board, self.score, self.is_game_over(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        new_board, score_gained, moved = move_board(self.board, action)
        if moved:
            self.board = new_board
            self.score += score_gained

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"Score: {self.score}"
        if action is not None:
            title += f" | Action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def is_move_legal(self, action):
        return is_move_possible(self.board, action)

    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))


