import gym
import universe  # register the universe environments
import numpy as np
import pdb
import pandas
import sklearn
import random
from keras.models import Sequential
from keras.layers import Dense

# Notes:
# Observation only has text which is blank and vision under observation_n[0] dict
# shape of observation vision: (768, 1024, 3)
# env.action_space.sample() gives a random action from those available, applies to any env

# Next step:
# need to investigate how i setup a simple model that is capable of reinforcement learning. e.g. RNN or LTSM but preferably simpler

def generateTrainingData():
    env = gym.make('CartPole-v0')
    #env.configure(remotes=1)  # automatically creates a local docker container
    goal_steps = 500
    score_requirement = 40
    initial_games = 10000
    x_train = []
    y_train = []
    scores = []
    y_train = []
    for i in range(initial_games):
        game_memory = []
        reward_n = 0
        game_score = 0
        observation_n = env.reset()

        for _ in range(goal_steps):
            action_n = chooseAction(observation_n, reward_n)
            previous_observation_n = observation_n
            observation_n, reward_n, done_n, info = env.step(action_n)
            game_memory.append([observation_n,action_n, reward_n, previous_observation_n])
            game_score += reward_n
            env.render()
            if done_n:
                print("Game: ",i," complete, score: ", game_score)
                break
        scores.append(game_score)
        if game_score > score_requirement:
            y_train.append(game_score)
            x_train.append(game_memory)

    x_train_array = np.array(x_train)
    y_train_array = np.array(y_train)
    np.save('x_train.npy',x_train_array)
    np.save('y_train.npy',y_train_array)
    if len(y_train) > 0:
        print("Average accepted score: ",np.mean(y_train) )
    else:
        print("No games met minimum score of: ",score_requirement)


def chooseAction(observation_n, reward_n):
    #if (reward_n != [0]):
    #    print (("Rewards: {}").format(reward_n))
    #if (observation_n != [None]):
        #screenshot(observation_n)
    action_n = random.randint(0,1)
    return action_n

def createModel():
    model = Sequential()
    model.add(Dense(64,activation='relu',input_shape=(4,) ))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

def trainModel(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    model.fit(x_train, y_train, epochs=20, batch_size=20, verbose=1)
    return model


def main():
    #generateTrainingData()
    model = createModel()
    trainModel(model)
    return

if __name__ == "__main__":
    main()
