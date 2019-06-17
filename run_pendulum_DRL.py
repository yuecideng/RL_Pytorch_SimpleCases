import gym
from agent.DQN import DeepQNetwork
import numpy as np 
import argparse
import matplotlib.pyplot as plt


def train(RL, env):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     
        
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > RL.batch_size:   # learning
            RL.learn()

        if total_steps - RL.batch_size > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1

    RL.plot_Q_value('Q_value_figure', 'DQN')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--double', default=False)
    parser.add_argument('--prioritized', default=False)
    parser.add_argument('--dueling', default=False)
    args = parser.parse_args()

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    ACTION_SPACE = 11
    agent = DeepQNetwork(ACTION_SPACE, 3, use_double_q=args.double, 
    prioritized=args.prioritized, dueling=args.dueling)
    train(agent, env)
