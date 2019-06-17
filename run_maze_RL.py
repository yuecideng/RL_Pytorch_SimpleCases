from environment.maze_env import Maze
from agent.QLearn import QLearn
from agent.Sarsa import Sarsa
import os

def update():
    for episode in range(1):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print(observation)
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
    env = Maze()
    RL = QLearn(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()