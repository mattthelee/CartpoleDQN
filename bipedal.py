import gym
import universe  # register the universe environments
import numpy as np
import json
import pdb
import pandas
import sklearn
import random

# Notes:
# Observation only has text which is blank and vision under observation_n[0] dict
# shape of observation vision: (768, 1024, 3)

# Next step:
# need to investigate how i setup a simple model that is capable of reinforcement learning. e.g. RNN or LTSM but preferably simpler
def main():
    env = gym.make('BipedalWalker-v2')
    #env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()
    reward_n = 0

    while True:
        action_n = chooseAction(observation_n, reward_n)
        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()

def screenshot(observation_n):
    np.savetxt("r.txt",observation_n[0]['vision'][0])
    np.savetxt("g.txt",observation_n[0]['vision'][1])
    np.savetxt("b.txt",observation_n[0]['vision'][2])

#def createModel():


def chooseAction(observation_n, reward_n):
    if (reward_n != [0]):
        print (("Rewards: {}").format(reward_n))
    #if (observation_n != [None]):
        #screenshot(observation_n)
    # hip_1, knee_1, hip_2, knee_2
    #action_n = [1,1,1,1]
    action_n = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
    return action_n

if __name__ == "__main__":
    main()
