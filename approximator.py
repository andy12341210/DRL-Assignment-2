import numpy as np
import matplotlib.pyplot as plt
from env2048 import Game2048Env
import dill
import copy
import os
import random
import math
from tqdm import tqdm
from collections import defaultdict
PERFECT_SNAKE = [[0, 0, 0, 1/2**12],
                [1/2**8, 1/2**9, 1/2**10, 1/2**11],
                [1/2**7, 1/2**6, 1/2**5, 1/2**4],
                [1,1/2,1/2**2,1/2**3]]

class NTupleApproximator:
    def __init__(self, board_size, patterns, init_value):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.init_value = init_value
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights1 = []
        # Generate symmetrical transformations for each pattern
        self.symmetry_groups1 = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups1.append(syms)
            self.weights1.append(defaultdict(float))
            # self.weights1.append(defaultdict(lambda : init_value))

        self.weights = []
        self.symmetry_groups = []
        for i in range(1):
           self.weights.append(copy.deepcopy(self.weights1))
           self.symmetry_groups.append(copy.deepcopy(self.symmetry_groups1))
        
        # print(self.weights)
        # print(f"symmetric_groups_size: {len(self.symmetry_groups)}, {len(self.symmetry_groups[0])}, {len(self.symmetry_groups[0][0])}")
        
        self.weights1 = []
        self.symmetry_groups1 = []
       
    def rotate_90(self, pattern):
      return [(y, self.board_size - 1 - x) for (x, y) in pattern]

    def flip_horizontal(self, pattern):
      return [(x, self.board_size - 1 - y) for (x, y) in pattern]

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        # Define rotations
        rotations = [pattern]
        for _ in range(3):  # Generate 3 more rotations
            pattern = self.rotate_90(pattern)
            rotations.append(pattern)
        symmetries = set(tuple(p) for p in rotations)  # use set to remove duplicates
        for rot in rotations:
            flipped = self.flip_horizontal(rot)
            symmetries.add(tuple(flipped))

        return [list(s) for s in symmetries]

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x][y]) for (x, y) in coords)

    def value(self, board, stage):
        total_value = 0
        for sym_group, weight in zip(self.symmetry_groups[stage], self.weights[stage]):
            group_value = 0
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                group_value += weight[tuple(feature)]
            group_value /= len(sym_group)
            total_value += group_value

        total_value /= len(self.symmetry_groups[stage])

        return total_value

    def update(self, board, delta, alpha, stage):
        TD_error = alpha*delta
        # print(TD_error)
        for sym_group, weight in zip(self.symmetry_groups[stage], self.weights[stage]):
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                weight[tuple(feature)] += TD_error

    def value_shaping(self, board):
        # return 0
        shaping_value = [0,0,0,0]
        for i in range(4):
            for j in range(4):
                shaping_value[0] += board[i][j] * PERFECT_SNAKE[i][j]
                shaping_value[1] += board[3-j][i] * PERFECT_SNAKE[i][j]
                shaping_value[2] += board[3-i][3-j] * PERFECT_SNAKE[i][j]
                shaping_value[3] += board[j][3-i] * PERFECT_SNAKE[i][j]

        return np.max(shaping_value)

def td_learning(env, approximator, num_episodes=10000, gamma=0.99,
                            alpha_arr = [0.4, 0.2, 0.01 , 0.0025],
                            epsilon_arr = [0,0,0,0],
                            name = None):
    
    def compute_afterstate(env, action):
        cloned_env = copy.deepcopy(env)
        cloned_env.move(action)
        return cloned_env.board, cloned_env.score - env.score

    def find_best_action(env, approximator, stage):
        best_value = -np.inf
        best_action = None
        for a in legal_actions:
            afterstate, reward = compute_afterstate(env, a)
            value = reward + gamma * approximator.value(afterstate, min(0,stage)) +approximator.value_shaping(afterstate)
            if value > best_value:
                best_value = value
                best_action = a
        return best_action
    
    save_dir = "./picture"
    filename = name+".png"
    save_path = os.path.join(save_dir, filename)

    final_scores = []
    success_flags = []

    max_tile = 0
    episode_max_tile = [0 for i in range(15)]

    epsilon_arr = epsilon_arr

    alpha = alpha_arr[0]
    epsilon = epsilon_arr[0]

    progress_bar = tqdm(range(num_episodes))

    EG_flag = 512

    for episode in progress_bar:

        if episode >= num_episodes*2/3:
            alpha = alpha_arr[3]
            epsilon = epsilon_arr[3]
        elif episode >= num_episodes/2:
            alpha = alpha_arr[2]
        elif episode >= num_episodes/3:
            epsilon = epsilon_arr[2]
        elif episode >= num_episodes/5:
            alpha = alpha_arr[1]
            epsilon = epsilon_arr[1]
        
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        done = False
        max_tile = np.max(state)
        stage = 0

        while not done:
            if max_tile >= 1024:
               stage = 1

            legal_actions = [a for a in range(4) if env.is_move_legal(a)]

            if not legal_actions:
              done = True
              break
            
            if np.random.rand()<=epsilon and max_tile >= EG_flag:
                action = np.random.choice(legal_actions)
            else:
                action = find_best_action(env, approximator, stage)

            # prev_score = env.score
            afterstate, reward = compute_afterstate(env, action)
            # current_state = copy.deepcopy(env.board)
            _, _, _, _ = env.step(action)
            max_tile = np.max(env.board)

            trajectory.append({
                # 'state': current_state,
                'afterstate': afterstate,
                'reward': reward,
                'stage': stage
            })

        # if episode == 0:
        #     print(trajectory)

        for t in reversed(range(len(trajectory))):
            # state = trajectory[t]['state']
            afterstate = trajectory[t]['afterstate']
            s = trajectory[t]['stage']
            reward = trajectory[t]['reward']

            # for i in range(s+1):
            # if i == 0: continue

            if t == len(trajectory)-1:
                next_value = 0
            else:
                next_afterstate = trajectory[t+1]['afterstate']
                next_value = approximator.value(next_afterstate, 0)
            
            value = approximator.value(afterstate, 0)
            td_error = reward + gamma * next_value - value

            # print(f"next_value: {next_value}")
            # print(f"value: {value}")
            # print(f"reward: {reward}")

            approximator.update(afterstate, td_error, alpha, 0)


        final_scores.append(env.score)

        for i in range(15):
            if max_tile == 2**i:
                episode_max_tile[i] += 1

        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            tqdm.write(f'Avg Score: {avg_score:.2f}, Success Rate: {success_rate:.2f}, Alpha: {alpha:.4f}, episode: {episode+1}')
            tqdm.write(str(episode_max_tile))
            # print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {a: {alpha:
            # print(episode_max_tile)
            if episode_max_tile[int(math.log(EG_flag,2))-1] <= 3 and episode_max_tile[int(math.log(EG_flag,2))-2] <= 1:
                EG_flag *= 2
                print(f"EG_flag :{EG_flag}")

            episode_max_tile = [0 for i in range(15)]

            window_size = 50
            moving_avg = np.convolve(final_scores, np.ones(window_size)/window_size, mode='valid')

            plt.figure(figsize=(12, 6))
            plt.plot(final_scores, alpha=0.3, label='Raw Scores')

            plt.plot(range(window_size-1, len(final_scores)), moving_avg,
                    color='red', linewidth=2, label=f'{window_size}-Episode Moving Avg')

            plt.xlabel('Episode')
            plt.ylabel('Final Score')
            plt.title('Training Progress with Moving Average')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()

        if (episode + 1) % 2000 == 0:
            with open(name+'.pkl', 'wb') as f:
                dill.dump(approximator, f)

    return final_scores


patterns = [
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    [(2,0), (2,1), (2,2), (2,3), (3,0), (3,1)],
    [(1,0), (2,0), (3,0), (1,1), (2,1), (3,1)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (2,0)],
    [(0,0), (1,0), (1,1), (2,1), (2,2), (2,3)]
]
