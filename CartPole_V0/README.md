This repository is a guide to automate the Google's Open AI game Cartpole_v0 using Deep Q learning method under the framework of Tensorflow. Complete guide is still under development phase.
Anyone who wants to contribute is welcomed here. 



# Introduction: #

Google's open AI gym module provides the Cartpole environment. It consists of a cart(shown in black color) and a vertical bar attached to the cartusing passive pivot joint. The cart can move left or right.The problem is to prevent the vertical bar from falling bymoving the car left or right. Below is the gif of the system:

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gym_animation.gif" width="350" height="250" />

### State vector of the system is a four dimensional vector having components: ###

- The position of the cart along the one-dimensional horizontal axis
- The cart’s velocity
- The angle of the pole from verticle
- The pole’s angular velocity

### Action space consists of two parameters: ###
- 0 (left movement of cart)
- 1 (right movement of cart)

So the cart can move either left or right in order to keep the pole balanced vertically.



# Rules: #

Maximum score or episode length in this system is 200. The game terminates if:

- The pole angle is more than ±12 degrees from thevertical axis.
- The cart position is more than ±2.4cm from the centre.
- The episode length is greater than 200.

The agent recieves the reward (score) 1 for being balanced. 

# Aim #
Our aim is to keep the pole vertically balanced until the score of 200 is reached. This problem can be cosidered solved if we achieve the average score of 195+ over the  course 100 consecutive episodes.


# Implementation of DQN algorithm #

I have used the Q learning algorithm to train, reward and penalize the model. It is a model free approach. Unlike Q-learning algorithm, we do not have to creat a table for remebering the actions and their rewards in certain state of the agent. In DQN we try to train a nueral network to predict the actions for the agent to perform. Here are the steps to train the model:

- Initially, the NN-model predicts the action randomly, then agent performs that action. Actually the model predicts the Q-values of all the possible actions and agent performs action with higher Q-value. 
- Based on the performance of agent, we get new state of the agent with certain rewards for the previous action.
- Positive reward for maintaining the balance and negative for falling (may change in other cases).
- We also take account of future actions while assigning positive reward to the model. For this, we use the model again to predict the action for the new state and based on that get the Q-value(most probable new action), multiply it with certain learning rate (gamma) and add it to the reward.  
- We appned the new_state, reward, previous_state in a python deque of certain length.
- When the python deque completely filled with these datas, then we randomly select some datas and make a batch of certain batch_size.
- Then these datas in a batch are feeded to the neural network for training.
- The intuition behind giving the negative reward means that, we will neglect these actions for the particular situation. Since, the agent always perform action which has higher Q-value, negative reward will automatically decrease the Q-value for bad actions.
-  This whole scheme of training the model is called as training via replay memory. Updt          

