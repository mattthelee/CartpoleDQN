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
# Need a predict function that calculates the estimated q scores
# Need a recall function that pulls in random action observations, calulculates q score

def generateTrainingData():
    env = gym.make('CartPole-v0')
    #env.configure(remotes=1)  # automatically creates a local docker container
    goal_steps = 500
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
            action_n = chooseRndAction()
            previous_observation_n = observation_n
            observation_n, reward_n, done_n, info = env.step(action_n)
            game_memory.append([observation_n,action_n, reward_n, previous_observation_n])
            game_score += reward_n
            env.render()
            if done_n:
                print("Game: ",i," complete, score: ", game_score)
                break
        scores.append(game_score)

    x_train_array = np.array(game_memory)
    np.save('x_train.npy',x_train_array)

def predict(model,game_memories):
     q = model.predict(game_memories)
     return q

def chooseRndAction():
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

def replay(batchStart,batchStop,model):
    x_train = np.load('x_train.npy')
    predicted_q = predict(model,x_train[batchStart:batchStop])
    for i in range(batchStart,batchStop):
        state, action, reward, oldState = x_train[i]

def main():
    #generateTrainingData()
    model = createModel()
    trainModel(model)
    return

if __name__ == "__main__":
    main()
