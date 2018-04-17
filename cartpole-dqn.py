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


# Notes:
# Observation only has text which is blank and vision under observation_n[0] dict
# shape of observation vision: (768, 1024, 3)
# env.action_space.sample() gives a random action from those available, applies to any env

# Next step:
# Need a predict function that calculates the estimated q scores
# Need a recall function that pulls in random action observations, calulculates q score

def sendAgentToTrainingCamp(env, agent):
    goal_steps = 500
    initial_games = 10000
    batch_size = 32
    scores = []
    for i in range(initial_games):
        reward = 0
        game_score = 0
        state = env.reset()

        for _ in range(goal_steps):
            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            if done:
                print("Game: ",i ," complete, score: " , game_score)
                new_state = None
                break
            agent.memory.append([state,action, reward, new_state])
            game_score += reward
            env.render()
            state = new_state

            scores.append(game_score)
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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(64,activation='relu',input_shape=(4,) ))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        batchLen = len(batch)
        states = np.array([ x[0] for x in batch ])
        newStates = np.array([ x[3] for x in batch ])

        predicted_qs = self.model.predict(states)
        newPredicted_qs = self.model.predict(newStates)

        for i in range(batchLen):
            state, action, reward, newState = batch[i]
            predicted_q = predicted_qs[0][i]
            if newState is None:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.amax(newPredicted_qs[0][i])

            # predict returns an array of actions and their associated predicted q vals
            # I then want to update the action that was taken in this case,
            # with the correct q value
            prediction = predicted_qs[i]
            # the [0] is required because of the way the predictions are returned
            prediction[0][action] = target_q
            x_train[i] = state
            # This is actually an updated prediction
            y_train[i] = prediction

        model.fit(x_train,y_train, epochs=1, verbose =0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
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
