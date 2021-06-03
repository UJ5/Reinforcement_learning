This repository is a guide to automate the Google's Open AI game Cartpole_v0 using Deep Q learning method under the framework of Tensorflow. Complete guide is still under development phase.
Anyone who wants to contribute is welcomed here. 



# Introduction: #

Google's open AI gym module provides the Cartpole environment. It consists of a cart(shown in black color) and a vertical bar attached to the cartusing passive pivot joint. The cart can move left or right.The problem is to prevent the vertical bar from falling bymoving the car left or right. Below is the gif of the system:

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gym_animation.gif" width="500" height="350" />

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
- Based on the performance of agent, we get new state of the agent with certain reward for the previous action.
- Positive reward for maintaining the balance and negative for falling (may change in other cases).
- We also take account of future actions while assigning positive reward to the model. For this, we use the model again to predict the action for the new state and based on that get the Q-value(most probable new action), multiply it with certain learning rate (gamma) and add it to the reward.  
- We appned the new_state, reward, previous_state in a python deque of certain length.
- When the python deque completely filled with these datas, then we randomly select some datas and make a batch of certain batch_size.
- Then these datas in a batch are feeded to the neural network for training.
- The intuition behind giving the negative reward means that, we will neglect these actions for the particular situation. Since, the agent always perform action which has higher Q-value, negative reward will automatically decrease the Q-value for bad actions.
-  This whole scheme of training the model is called as training via replay memory.         


### Basic Q-learning algorithm : ###
```
initialize deque, EPSILON, learning_rate(gamma)
for episode in EPISODES:
    state = env.reset()                                                      # reset the environment
    while not done:
        if(EPSILON>=random_value):                                           # 0<= random_value <=1
            action = random(action_space)                                    # perform any action randomly from action  
        else:
            action = max(model.predict(state))
        new_state, reward, done, _ = env.step(action)            # agent performs the action using step() function and we get new state, reward and a boolean
        if not done:
            reward = reward + gamma*[max(model.predict(new_state))]
        else :
            reward = -10
        deque.append([state,reward, new_state, action, done])
        
        state = new_state
       
        if(length(deque)>=MINIBATCH):                                         # training the model
            minibatch = random.sample(deque, batch_size)
            model.fit(minibatch)
            EPSILON *= DECAY_RATE
            
```

# Exploitation vs Exploration #

Initially, the agent explores the environment more by performing some random actions. This is called *exploration*. As the learning proceeds, the agent performs action more based on prediction from the NN model. This is called *exploitation*. So, how do we ensure exploration is decreasing and exploitation is increasing throughout the learning? I have used the *Epsilon-greedy method* for this. How:

- I initialized a variable EPSILON = 1 and a DECAY_RATE as 0.995.
- As the training occurs, the value of epsilon decays as EPSILON*DECAY_RATE. 
- As the EPSILON decreases, the agent shifts from random exploration to exploitation based on training of NN model. 
  
    
    
       
#### Below graph shows the variation of EPSILON per episode: ####

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/graphs/epsilon_vs_episodes.png" width="550" height="350" />
       
         
           
           
#### Below graph shows how number of exploration/exploitation varies per episode: ####

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/graphs/exploitation-exploration_vs_episodes.png" width="550" height="350" />
     
       
         
           
           
#### Below graph shows the variation of score per episode: ####
 
<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/graphs/score_vs_episodes.png" width="550" height="350" />
       
     
            
                   
                       

#### Below graph shows the variation of average_score per episode: ####
<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/graphs/avg-score_vs_episode.png" width="550" height="350" />



## Performance of agent after several interval of trainings: ##
(hover your pointer on screens to see the number of episodes)

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/50_episodes.gif" width="300" height="180" alt = "50_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/100_episodes.gif" width="300" height="180" title = "100_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/150_episodes.gif" width="300" height="180" title = "150_episodes" />

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/200_episodes.gif" width="300" height="180" title = "200_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/250_episodes.gif" width="300" height="180" title = "250_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/300_episodes.gif" width="300" height="180" title = "300_episodes" />

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/350_episodes.gif" width="300" height="180" title = "350_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/400_episodes.gif" width="300" height="180" title = "400_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/500_episodes.gif" width="300" height="180" title = "500_episodes" />

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/700_episodes.gif" width="300" height="180" title = "700_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/900_episodes.gif" width="300" height="180" title = "900_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/1100_episodes.gif" width="300" height="180" title = "1100_episodes" />   

<img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/1300_episodes.gif" width="300" height="180" title = "1300_episodes" />   <img src="https://github.com/UJ5/Reinforcement_learning/blob/main/CartPole_V0/gifs/1950_episodes.gif" width="300" height="180" title = "1950_episodes" />  

After the end of 1950th episode training, the agent (pole) is quite able to balance itself using the trained model. Although the model requires the hyperparameter tuning, so that it can quickly using in less number of episodes. 

*Note: I am curently working on hyperparameter tuning to make it more optimal. Anyone who wants to contribute is welcomed here :)
