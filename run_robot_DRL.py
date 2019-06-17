from agent.DDPG import DDPG
from agent.TD3 import TD3
from environment.arm_env import ArmEnv
import numpy as np 
import matplotlib.pyplot as plt
from math import pi
import torch
import shutil
import argparse

np.random.seed(1)
#Traning hyperparameters 
TRAIN_CONFIG = {'state_dim':7,'action_dim':2,'action_bound':1,'train_epoch':300,'train_step':200,
                'pre_trained':False,'cuda':False}

def main(args):
    env = ArmEnv(mode='hard')
    if args.algorithm == 'TD3': model = TD3(a_dim=TRAIN_CONFIG['action_dim'],s_dim=TRAIN_CONFIG['state_dim'])
    if args.algorithm == 'DDPG': model = DDPG(a_dim=TRAIN_CONFIG['action_dim'],s_dim=TRAIN_CONFIG['state_dim'])
    '''       
    #load pre_trained model        
    if TRAIN_CONFIG['pre_trained']:
        model.load_model()
    '''
    total_reward_list = []
    for epoch in range(TRAIN_CONFIG['train_epoch']): 
        state = env.reset()
        total_reward = 0
        for i in range(TRAIN_CONFIG['train_step']):
            env.render()
            action = model.choose_action(state)
            state_, reward, terminal = env.step(action)
            model.store_transition(state,action,reward,state_,terminal)
            state = state_
            total_reward += reward

            if model.memory_counter > 1000:
                model.Learn()            

        total_reward_list.append(total_reward)
        print('epoch:', epoch,  '||',  'Reward:', total_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='TD3')
    args = parser.parse_args()
    assert args.algorithm == 'TD3' or args.algorithm == 'DDPG', 'Please type TD3 or DDPG'
    main(args)
