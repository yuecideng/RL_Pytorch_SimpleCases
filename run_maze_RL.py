from environment.maze_env import Maze
from agent.QLearn import QLearn
from agent.Sarsa import Sarsa
import argparse
import os

def update():
    for episode in range(1000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
                
    RL.save_table('Q_table','Q-learning')
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='Q-learning')
    args = parser.parse_args()

    assert args.algorithm == 'Q-learning' or args.algorithm == 'Sarsa', 'Please type Q-learning or Sarsa'
    if args.algorithm == 'Q-learning': RL = QLearn(actions=list(range(env.n_actions))) 
    if args.algorithm == 'Sarsa': RL = Sarsa(actions=list(range(env.n_actions)))
        
    env = Maze()
    env.after(100, update)
    env.mainloop()