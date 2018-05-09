This repo is for my experiments with open-ai's Gym. It is a combination of reinforcement learning and deep learning to solve these problems. 

## Cartpole

The cartpole problem involves balancing a pole that is connected to a cart by a hinge. To keep the pole balanced the agent must decide whether to move the cart to the right or not. The open-ai scenario does not allow the cart to stay still which adds additional challenge. The choice to move left or right is given at eash step of the simulation. The simulation gives a reward of 1 for each step that the pole remains balanced. The simulation finishes a given game when the pole moves more than 15 degrees from center, or the cart moves more than 2.4 units from the center. The problem is considered solved when the last 50 games had an average score over 200. Further details can be found here: https://gym.openai.com/envs/CartPole-v1/

# Solution

To solve the cartpole problem I used a q-learning network. The q-learning network uses a Neural Network to predict the q-value of a given course of action. This q-value is then updated using the Bellman equation. The algorithm begins by taking random actions and recording the rewards granted. Over time the algorithm is increasingly likely to choose the action with the highest q-value rather than taking a random action.  
