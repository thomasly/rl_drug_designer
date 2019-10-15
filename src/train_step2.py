import os
import pickle as pk
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
from utils.model_actions import make_name
from utils.mylog import MyLog


class DQNAgent:
    def __init__(self, model_path, lr=1e-4, state_size=100, action_size=130):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # decay or discount rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01 # minimum amount of random exploration
        self.learning_rate = lr
        self.model = self._get_model(model_path)
    
    def _get_model(self, model_path):
        # neural net to approximate Q-value function:
        model = tf.keras.models.load_model(model_path)
        # recompile the model, use MSE as loss
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # if acting randomly, take random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # if not acting randomly, predict reward value based on current state
        act_values = self.model.predict(state)
        # pick the next token that will give the highest reward
        return np.argmax(act_values[0]) 

    def replay(self, batch_size):
        """ method that trains NN with experiences sampled from memory
        """
        # sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
         # extract data for each minibatch sample
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done, then target = reward
            if not done: # if not done, then predict future discounted reward
                # (target) = reward + (discount rate gamma) * 
                # (maximum target Q based on future action a')
                target = (reward + self.gamma * 
                          np.amax(self.model.predict(next_state)[0])) 
            # approximately map current state to future discounted reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # single epoch of training with x=state, y=target_f; 
            # fit decreases loss btwn target_f and y_hat
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train():
    state_size = 100
    action_size = 130
    agent = DQNAgent(state_size, action_size) # initialise agent
    done = False
    # iterate over new episodes
    for e in range(n_episodes):
        # reset state at start of each new episode
        state = np.zeros((1, state_size, action_size))
        
        for i in range(state_size):
            action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
            next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
            reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
            state = next_state # set "current state" for upcoming iteration to the current next state        
            if done: # episode ends if agent drops pole or we reach timestep 5000
                print("episode: {}/{}, score: {}, e: {:.2}" # print the episode's score and agent's epsilon
                    .format(e, n_episodes, time, agent.epsilon))
                break # exit loop
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
