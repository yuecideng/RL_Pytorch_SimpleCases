import numpy as np
import pickle
import os


class Sarsa:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def choose_action(self, s, explore=True):
        s = str(s)
        # action selection
        if explore:
            if np.random.uniform() < self.epsilon:
                q = np.array([self.q_table.get((s, a), 0.0) for a in self.actions])
                action = np.random.choice(np.argwhere(q==np.max(q)).flatten())
            else:
                action = np.random.choice(self.actions)
        else:
            q = np.array([self.q_table.get((s, a), 0.0) for a in self.actions])
            action = np.random.choice(np.argwhere(q==np.max(q)).flatten())

        return action

    def learn(self, s, a, r, s_):
        s, s_ = str(s), str(s_)
        a_ = self.choose_action(s_)
        q_target = r + self.gamma * self.q_table.get((s_, a_), 0.0)
        q_predict = self.q_table.get((s, a), None)
        if q_predict is None:
            self.q_table[(s, a)] = r # next state is terminal
        else:
            self.q_table[(s, a)] = q_predict + self.lr * (q_target - q_predict)  # next state is not terminal

    def save_table(self,model_dir,model_name):
        if not os.path.exists(model_dir+'/'+model_name):
            os.makedirs(model_dir+'/'+model_name)
        with open(model_dir+'/'+model_name+'/'+'/QLearn_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f, pickle.HIGHEST_PROTOCOL)

    def load_table(self,model_dir,model_name):  
        with open(model_dir+'/'+model_name+'/'+'/QLearn_table.pkl', 'rb') as f:
            self.q_table = pickle.load(f)