from env2048 import Game2048Env
# from evaluate import evaluate
# from Approximator import NTupleApproximator, patterns, PERFECT_SNAKE
import copy
import random
import math
import dill
import numpy as np

import numpy as np
import random
from env2048 import Game2048Env, move_board
from evaluate import evaluate, create_env_from_state
PERFECT_SNAKE = [[0, 0, 0, 1/2**12],
                [1/2**8, 1/2**9, 1/2**10, 1/2**11],
                [1/2**7, 1/2**6, 1/2**5, 1/2**4],
                [1,1/2,1/2**2,1/2**3]]



# approximator = NTupleApproximator(board_size=4, patterns=patterns, init_value=0)
# with open('model.pkl', 'rb') as f:
#     approximator = dill.load(f)
approximator = None

def value_shaping(board):
    #return 0
    shaping_value = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            shaping_value[0] += board[i][j] * PERFECT_SNAKE[i][j]
            shaping_value[1] += board[3-j][i] * PERFECT_SNAKE[i][j]
            shaping_value[2] += board[3-i][3-j] * PERFECT_SNAKE[i][j]
            shaping_value[3] += board[j][3-i] * PERFECT_SNAKE[i][j]

    return np.max(shaping_value)*2048


class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def compute_afterstate(self, env, action):
        cloned_env = copy.deepcopy(env)
        old_score = cloned_env.score
        cloned_env.move(action)
        return cloned_env.board, cloned_env.score - old_score
    
    def greedy_action(self, board, actions):
        best_value = -np.inf
        best_action = None
        new_env = self.create_env_from_state(board, 0)
        for a in actions:
            afterstate, reward = self.compute_afterstate(new_env, a)
            value = reward + value_shaping(afterstate)
            if value > best_value:
                best_value = value
                best_action = a
        return best_action

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.

        best_value = -float('inf')
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                q_value = float('inf')
            else:
                q_value = child.total_reward / child.visits

            exploration = self.c * np.sqrt(np.log(node.visits + 1) / (child.visits + 1e-5))
            uct_value = q_value + exploration
            # print(f"q_value: {q_value}, exploration: {exploration}")

            if uct_value > best_value:
                best_value = uct_value
                best_action = action

        return best_action

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        total_reward = 0.0
        current_gamma = 1.0
        episilon = 0.2
        for _ in range(depth):

            if sim_env.is_game_over():
                break
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            if np.random.rand() <= episilon:
                action = np.random.choice(legal_actions)
            else:
                action = self.greedy_action(sim_env.board, legal_actions)
                
            old_score = sim_env.score
            sim_env.step(action)
            step_reward = sim_env.score - old_score
            total_reward += step_reward * current_gamma
            current_gamma *= self.gamma

        # print(f"total reward : {total_reward}")

        if not sim_env.is_game_over():
          # total_reward += self.approximator.value(sim_env.board,0) * current_gamma
          total_reward += value_shaping(sim_env.board) * current_gamma

        # print(f"total reward with value : {total_reward}")

        return total_reward


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            action = self.select_child(node)
            node = node.children[action]
            sim_env.step(action)


        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            sim_env.step(action)
            child_node = TD_MCTS_Node(sim_env.board.copy(), sim_env.score, node, action)
            node.children[action] = child_node
            node.untried_actions.remove(action)
            node = child_node


        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        sim_env = create_env_from_state(root.state,root.score)
        best_action = evaluate(sim_env, approximator, 3)
        return best_action, distribution

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
    iterations = [0,0,0,0]

    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=3, gamma=0.9)

    iteration = iterations[0]

    if np.max(env.board) >= 2048:
        iteration = iterations[2]
    elif np.max(env.board) >= 1024:
        iteration = iterations[1]

    root = TD_MCTS_Node(state, env.score)

    for _ in range(iteration):
        td_mcts.run_simulation(root)

    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act
