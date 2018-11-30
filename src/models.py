import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.utils import * 
from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
from keras.optimizers import Adam
from tqdm import tqdm
import sys
from collections import deque

class Speaker():
    def __init__(self,state_size,class_size,action_size,bias,seed):
        self.gamma = 0
        self.epsilon = -1
        self.decay = 0.99
        self.state_size = state_size
        self.class_size = class_size
        self.action_size = action_size
        self.seed = seed
        self.model = self.initialize_model(bias)
        self.params = {"states":[],"class_states":[],"actions":[],"action_probs":[],"rewards":[]}
        self.episode_rewards = []
        
    def initialize_model(self,bias):
        inp = Input(shape=(self.state_size,))
        dense3 = Dense(32,activation="relu",kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(inp)
        output = Dense(self.action_size,kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),activation="linear")(dense3)
        output = Softmax()(output)
        model = Model(inputs=[inp],outputs=output)
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=(0.001*bias)))
        return model
    
    def act(self,state):
        if np.random.random() < self.epsilon:
            self.epsilon *= self.decay
            return np.random.choice(self.action_size) , np.zeros(self.action_size) 
        policy = self.model.predict([state]).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy
    
    def save(self,state,action,probs,reward):
        self.params["states"].append(state)
        self.params["actions"].append(action)
        self.params["action_probs"].append(probs)
        self.params["rewards"].append(reward)
    
    def reset(self):
        self.params["states"] = []
        self.params["class_states"] = []
        self.params["actions"] = []
        self.params["action_probs"] = []
        self.params["rewards"] = []
        
    def train(self):
        episode_length = len(self.params["states"])
        rewards = np.array(self.params["rewards"])
        states = np.zeros((episode_length,self.state_size))
        actions = np.zeros((episode_length,self.action_size))
        action_probs = np.zeros((episode_length,self.action_size))
        for i in range(episode_length):
            state = self.params["states"][i]
            action = np.zeros((1,self.action_size))
            action[0][self.params["actions"][i]] = rewards[i]
            loss = self.model.fit(state, action,batch_size=32, epochs=1, verbose=0)
        self.reset()

class Listener():
    
    def __init__(self,state_size,vocab_size,action_size,bias,seed):
        self.gamma = 0
        self.epsilon = -1
        self.decay = 0.99
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.action_size = action_size
        self.seed = seed
        self.model = self.initialize_model(bias)
        self.params = {"states":[],"vocab_states":[],"actions":[],"action_probs":[],"rewards":[]}
        self.episode_rewards = []

    def append_dense(self,inp,dense):
        dense = dense(inp)
        return dense
    
    def initialize_model(self,bias):
        model_dict = {}
        input_list = []
        dense_list = Dense(32,activation="relu",kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))
        for i in range(self.action_size):
            model_dict["input_" + str(i)] = Input(shape=(self.state_size,))
            input_list.append(model_dict["input_" + str(i)])
            model_dict["dense_set" + str(i)] = self.append_dense(model_dict["input_" + str(i)],dense_list)
        label_inp = Input(shape=(self.vocab_size,))
        input_list.append(label_inp)
        dense_label = Dense(32,activation="relu",kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(label_inp)
        dot_list = []
        for i in range(self.action_size):
            dot_val = dot([dense_label,model_dict["dense_set" + str(i)]],axes=1)
            dot_list.append(dot_val)
        output = concatenate(dot_list)
        output = Softmax()(output)
        model = Model(inputs=input_list,outputs=output)
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=(0.001*bias)))
        return model
    
    def act(self,state):
        if np.random.random() < self.epsilon:
            self.epsilon *= self.decay
            return np.random.choice(self.action_size) , np.zeros(self.action_size)     
        policy = self.model.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy
    
    def save(self,state,vocab_state,action,probs,reward):
        self.params["states"].append(state)
        self.params["vocab_states"].append(vocab_state)
        self.params["actions"].append(action)
        self.params["action_probs"].append(probs)
        self.params["rewards"].append(reward)
        
    def reset(self):
        self.params["states"] = []
        self.params["vocab_states"] = []
        self.params["actions"] = []
        self.params["action_probs"] = []
        self.params["rewards"] = []
        
    def train(self):
        episode_length = len(self.params["states"])
        rewards = np.array(self.params["rewards"])
        states = np.zeros((episode_length,self.state_size))
        vocab_states = np.zeros((episode_length,self.vocab_size))
        action_probs = np.zeros((episode_length,self.action_size))
        actions = np.zeros((episode_length,self.action_size))
        for i in range(episode_length):
            state = self.params["states"][i]
            action = np.zeros((1,self.action_size))
            action[0][self.params["actions"][i]] = rewards[i]
            loss = self.model.fit(state, action,batch_size=32, epochs=1, verbose=0)
        self.reset()
