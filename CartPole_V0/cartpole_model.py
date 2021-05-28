import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from collections import deque
import numpy as np
import random
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


EPISODES = 2000
output_dir = "/home/ug2018/mst/18114017/ML/"
EPSILON = 1
REPLAY_MEMORY = deque(maxlen=400)
MIN_EPSILON = 0.01
DECAY_RATE = 0.995
MINIBATCH = 350
GAMMA = 0.99

env = gym.make('CartPole-v0')
env.seed(0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


class DQNagent:

    def __init__(self):

        self.fit_model = self.create_model()

        self.predict_model = self.create_model()
        self.predict_model.load_weights("predict_weights_1100.hdf5")

        self.targets = []
        self.states = []

    def create_model(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, activation ="relu",input_dim = state_size))
        model.add(tf.keras.layers.Dense(32, activation ="relu"))
        model.add(tf.keras.layers.Dense(action_size, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
        return model

    def model_summary(self,model):
        return model.summary()

    def get_q(self, state):
        return self.predict_model.predict(state)

    def train(self,batch_size): 
        minibatch = random.sample(REPLAY_MEMORY, batch_size)
        for state, reward, action, new_state, done in minibatch:
            if done :
                target = reward 
            else:
                target = reward + GAMMA * np.amax(self.get_q(new_state)[0])
            target_f = self.get_q(state) 
            target_f[0][action] = target

            self.states.append(state[0])
            self.targets.append(target_f[0])

        self.fit_weights(self.states,self.targets)
	
  
    def fit_weights(self, states, targets):
        self.fit_model.fit(np.array(states), np.array(targets), batch_size = MINIBATCH, epochs = 5 ,verbose=0)
        self.predict_model.set_weights(self.fit_model.get_weights())

    def predict_save(self, name): 
        self.predict_model.save_weights(name)
    def fit_save(self, name):    
        self.fit_model.save_weights(name)

    



agent = DQNagent()
print(agent.fit_model.summary())




x=[]
y=[]
z=[]

def update_graph(z,y):
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.plot(z,y)
    plt.pause(0.5)
plt.show()
    

for eps in range(EPISODES):
    env.reset()
    done = False
    state = env.reset()
    state = np.reshape(state, [1,state_size])
    time = 0
    exp=0
    elp=0
    while not done:
        env.render()
        #if EPSILON >= np.random.rand():
        #    exp +=1
        #    action = random.randrange(action_size) 
        #else:
        elp +=1
        action = np.argmax(agent.get_q(state)[0])
        new_state, reward, done, _ = env.step(action)
        new_state = np.reshape(new_state,[1, state_size])
        #reward = reward if not done else -100
        #REPLAY_MEMORY.append((state,reward,action,new_state,done))
        state = new_state
        time += 1
    env.close()
    #x.append([eps,exp,elp,time,EPSILON])
    y.append(time)
    z.append(eps)
    update_graph(z,y)
    #if (len(REPLAY_MEMORY)) >= MINIBATCH:
    #    agent.train(MINIBATCH)
    #    if EPSILON > MIN_EPSILON:
    #        EPSILON *= DECAY_RATE
    #if eps % 50 == 0:
    #    agent.predict_save(output_dir + "predict_weights_" + '{:04d}'.format(eps) + ".hdf5")
    #    agent.fit_save(output_dir + "fit_weights_" + '{:04d}'.format(eps) + ".hdf5")
    #with open("score_vs_eps.txt", "w") as output:
    #    output.write("Episodes"+"   "+"Exploration"+"   " + "Exploitation" + "  "+ "Score" + "  " + "Epsilon"+"\n")
    #    for eps,exp,elp,time,epsilon in x:
    #        output.write("      "+str(eps)+"        "+str(exp)+"        "+str(elp)+"        "+str(time)+"       "+"{:.4f}".format(epsilon) +"\n")

#agent.predict_model.save('CartPole_predict_model')
#agent.predict_model.save('CartPole_fit_model')

