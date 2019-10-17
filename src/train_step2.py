import os

import numpy as np

from utils.paths import Path
from utils.rl_agents import DQNAgent
from utils.smiles_reader import get_int2token
from utils.smiles_reader import is_valid


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
                print(smiles, end="")
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
