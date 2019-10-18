import os

import numpy as np

from utils.paths import Path
from utils.models import optimize_gpu_usage
from utils.rl_agents import DQNAgent
from utils.smiles_reader import get_int2token
from utils.smiles_reader import is_valid


def play_episode(agent):
    done = False
    # reset state at start of each new episode
    state = np.zeros((1, state_size, action_size))
    characters = list()
    memory = list()
    reward = 0.001
    index_to_char = get_int2token()
    for i in range(state_size):
        action = agent.act(state, i, greedy=True)
        next_state = state
        next_state[0, i+1, action] = 1
        if i == state_size - 2:
            character = '\n'
        else:
            character = index_to_char[action]
        characters.append(character)
        # change the reward value at the end of episode
        if character == "\n":
            done = True
            smiles = "".join(characters)
            if is_valid(smiles):
                reward = 0.9

        memory.append([state, action, reward, next_state, done, i])
        # set "current state" for upcoming iteration to the next state
        state = next_state
        # episode ends if "\n" is generated or reach the max SMILES length
        if done:
            break
    # change all the reward values in the memory
    if reward != 0:
        for m in memory:
            m[2] = reward
    return memory, smiles, reward


def train(agent, n_episodes=10, batch_size=32):
    # optimize gpu memory usage
    optimize_gpu_usage()

    output_dir = os.path.join(Path.checkpoints, "train_step2")
    os.makedirs(output_dir, exist_ok=True)
    # iterate over new episodes
    e = 0
    n_valid = 0
    while e < n_episodes:
        while n_valid < 10:
            memory, smiles, reward = play_episode(agent)
            print("SMILES: {}, reward: {}".format(smiles.strip(), reward))
            if reward != 0:
                n_valid += 1
                for m in memory:
                    agent.remember(*m)
        memory, smiles, reward = play_episode(agent)
        print("episode: {}/{}, SMILES: {}, reward: {}".format(
            e+1, n_episodes, smiles.strip(), reward))
        for m in memory:
            agent.remember(*m)

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
    parser.add_argument("-e", "--episodes", type=int, default=100,
                        help="Number of episodes. Infinite if 0.")
    args = parser.parse_args()
    state_size = 100
    action_size = 130
    model_path = os.path.join(
        Path.checkpoints, "weights-improvement-40-0.1822-bigger.hdf5")
    agent = DQNAgent(
        model_path,
        state_size=state_size, action_size=action_size)
    n_episodes = args.episodes
    if n_episodes == 0:
        n_episodes = float("inf")
    train(agent, n_episodes=n_episodes)
