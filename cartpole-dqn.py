import gym
import universe  # register the universe environments
import numpy as np
import pdb
import pandas
import sklearn
import random
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import pdb


# Notes:
# Observation only has text which is blank and vision under observation_n[0] dict
# shape of observation vision: (768, 1024, 3)
# env.action_space.sample() gives a random action from those available, applies to any env

def sendAgentToTrainingCamp(env, agent):
    goal_steps = 500
    initial_games = 10000
    batch_size = 32
    scores = deque(maxlen=50)
    for i in range(initial_games):
        if i > 50 and np.mean(scores) > 200:
            print("Problem solved at game: ", i - 1)
            break
        reward = 0
        game_score = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        for j in range(goal_steps):
            #print("Starting goal step: ", j, " of game: ", i, " avg score: ", np.mean(scores))
            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            new_state = np.reshape(new_state, [1, 4])
            agent.memory.append((state,action, reward, new_state, done))

            if done:
                print("Game: ",i ," complete, score: " , game_score," last 50 scores avg: ", np.mean(scores), " epsilon ", agent.epsilon)
                scores.append(game_score)
                break
            game_score += reward
            #env.render()
            state = new_state


            if len(agent.memory) > batch_size:
                randomBatch = random.sample(agent.memory, batch_size)
                agent.replay(randomBatch)
    return scores

class agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(64,activation='relu', input_dim=self.state_size))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        # Randomly choose to take a randomly chosen action to allow exploration
        # When epsilon is high, higher chance, therefore decrease it overtime
        # This then results in exploration early on with greater exploitation later
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch):
        x_train = []
        y_train = []
        for state, action, reward, newState, done in batch:
            if done:
                # Set the reward for finishing the game
                target_q = reward
            else:
                target_q = reward + self.gamma * np.amax(self.model.predict(newState)[0])
            prediction = self.model.predict(state)
            prediction[0][action] = target_q
            x_train.append(state[0])
            y_train.append(prediction[0])
        self.model.fit(np.asarray(x_train),np.asarray(y_train),epochs=1,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0
        return

def main():

    env = gym.make('CartPole-v1')
    #env.configure(remotes=1)  # automatically creates a local docker container

    # Get the number of available states and actions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    myagent = agent(state_size, action_size)
    scores = sendAgentToTrainingCamp(env, myagent)
    print (scores)
    return

if __name__ == "__main__":
    main()
