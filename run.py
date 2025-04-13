from env2048 import Game2048Env
from student_agent import get_action


def main():
    env = Game2048Env()
    env.reset()
    while not env.is_game_over():
        action = get_action(env.board,env.score)
        env.step(action)
        # env.render()

    print(env.score)
    return env.score

if __name__ == '__main__':
    score = []
    for i in range(10):
        score.append(main())
