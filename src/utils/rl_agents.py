from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np

from .models import lstm_model


class DQNAgent:
    def __init__(self, model_path=None, lr=1e-4, state_size=100,
                 action_size=130):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # decay or discount rate
        if model_path is None:
            self.epsilon = 1.0  # exploration rate: how much to act randomly
            self.epsilon_decay = 0.999
            self.epsilon_min = 0.01  # minimum amount of random exploration
        else:
            self.epsilon = 0.1  # exploration rate: how much to act randomly
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.001  # minimum amount of random exploration
        self.learning_rate = lr
        self.model = self._get_model(model_path)

    def _get_model(self, model_path):
        # neural net to approximate Q-value function:
        if model_path is None:
            model = lstm_model((100, 130))
        else:
            model = tf.keras.models.load_model(model_path)
        # recompile the model, use MSE as loss
        # model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, index):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done, index))

    def act(self, state, index, greedy=False):
        # if acting randomly, take random action
        if not greedy and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # if not acting randomly, predict reward value based on current state
        act_values = self.model.predict(state)[0, index]
        act_values = act_values / np.sum(act_values)
        # pick the next token
        return np.random.choice(range(self.action_size), p=act_values)

    def replay(self, batch_size, current_episode=None,
               update_epsilon_threshold=100):
        """ method that trains NN with experiences sampled from memory
        """
        # sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        # extract data for each minibatch sample
        for state, action, reward, next_state, done, i in minibatch:
            target = reward  # if done, then target = reward
            if not done:  # if not done, then predict future discounted reward
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
        if current_episode is None:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            if current_episode > update_epsilon_threshold and \
                                self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
