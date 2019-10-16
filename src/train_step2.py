import os
from collections import deque
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping

from models import lstm_model
from utils.paths import Path
from utils.smiles_reader import smiles_sampler
from utils.smiles_reader import smiles2sequence
from utils.smiles_reader import get_smiles_tokens
from utils.smiles_reader import get_token2int
from utils.smiles_reader import get_int2token
from utils.smiles_reader import is_valid
from utils.model_actions import make_name
from utils.mylog import MyLog


class DQNAgent:
    def __init__(self, model_path=None, lr=1e-4, state_size=100, action_size=130):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # decay or discount rate
        if model_path is None:
            self.epsilon = 1.0 # exploration rate: how much to act randomly
            self.epsilon_decay = 0.999
            self.epsilon_min = 0.01 # minimum amount of random exploration
        else:
            self.epsilon = 0.1 # exploration rate: how much to act randomly
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.001 # minimum amount of random exploration
        self.learning_rate = lr
        self.model = self._get_model(model_path)
    
    def _get_model(self, model_path):
        # neural net to approximate Q-value function:
        if model_path is None:
            model = lstm_model((100, 130))
        else:
            model = tf.keras.models.load_model(model_path)
        # recompile the model, use MSE as loss
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done, index):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done, index))

    def act(self, state, index):
        # if acting randomly, take random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # if not acting randomly, predict reward value based on current state
        act_values = self.model.predict(state)[0, index]
        # pick the next token that will give the highest reward
        return np.argmax(act_values) 

    def replay(self, batch_size):
        """ method that trains NN with experiences sampled from memory
        """
        # sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
         # extract data for each minibatch sample
        for state, action, reward, next_state, done, i in minibatch:
            target = reward # if done, then target = reward
            if not done: # if not done, then predict future discounted reward
                # (target) = reward + (discount rate gamma) * 
                # (maximum target Q based on future action a')
                target = (reward + self.gamma * 
                          np.amax(self.model.predict(next_state)[0, i+1])) 
            # approximately map current state to future discounted reward
            target_f = self.model.predict(state)
            target_f[0, i, action] = target
            # single epoch of training with x=state, y=target_f; 
            # fit decreases loss btwn target_f and y_hat
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train(agent, n_episodes=10, batch_size=32):
    
    index_to_char = get_int2token()
    output_dir = os.path.join(Path.checkpoints, "train_step2")
    os.makedirs(output_dir, exist_ok=True)
    # iterate over new episodes
    e = 0
    while e < n_episodes:
        done = False
        # reset state at start of each new episode
        state = np.zeros((1, state_size, action_size))
        name = list()
        for i in range(state_size):
            action = agent.act(state, i)
            next_state = state
            next_state[0, i, action] = 1
            if i == state_size - 2:
                character = '\n'
            else:
                character = index_to_char[action]
            name.append(character)
            
            if character == "\n":
                done = True
                smiles = "".join(name)
                if is_valid(smiles):
                    reward = 1
                else:
                    reward = -1
            else:
                reward = 0
            # remember the previous timestep's state, actions, reward, etc.
            agent.remember(state, action, reward, next_state, done, i)
            # set "current state" for upcoming iteration to the next state     
            state = next_state
            # episode ends if "\n" is generated or reach the max SMILES length      
            if done:
                smiles = "".join(name)
                print(smiles, end="")  ### delete this
                # print the episode's score and agent's epsilon
                print(
                    "episode: {}/{}, reward: {}, e: {:.2}".format(
                        e+1, n_episodes, reward, agent.epsilon))
                # exit loop
                break
        if len(agent.memory) > batch_size:
            # train the agent by replaying the experiences of the episode
            agent.replay(batch_size)

        if e + 1 % 50 == 0:
            agent.save(os.path.join(
                        output_dir, "weights_" + '{:04d}'.format(e) + ".hdf5"))
        e += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=100)
    args = parser.parse_args()
    state_size = 100
    action_size = 130
    model_path = os.path.join(
        Path.checkpoints, "weights-improvement-40-0.1822-bigger.hdf5")
    agent = DQNAgent(
        # model_path, 
        state_size=state_size, action_size=action_size)
    n_episodes = args.episodes
    if n_episodes == 0:
        n_episodes = float("inf")
    train(agent, n_episodes=n_episodes)
