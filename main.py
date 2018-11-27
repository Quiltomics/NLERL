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
from models import Speaker, Listener

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def generate_data(x_train,y_train,x_test,y_test):
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, 
                                       weights='imagenet', input_tensor=None, 
                                       input_shape=None, pooling='avg')
    unique_vals = np.unique(y_train)
    data_dict = {}
    data_dict["X_train"] = {}
    data_dict["X_test"] = {}
    bar = tqdm(np.arange(unique_vals.shape[0]))
    for i in bar:
        train_indices = np.where(y_train == i)[0]
        train_indices = np.random.choice(train_indices,200)
        test_indices = np.where(y_test == i)[0]
        test_indices = np.random.choice(test_indices,1)
        train_data = x_train[train_indices,:,:]
        test_data = x_test[test_indices,:,:]
        y_train_data = y_train[train_indices]
        y_test_data = y_test[test_indices]
        data_dict["X_train"][i] = vgg16.predict(train_data)
        data_dict["X_test"][i] = vgg16.predict(test_data)
    data_dict["y_train"] = np.squeeze(y_train)
    data_dict["y_test"] = np.squeeze(y_test)
    print("Finished Gathering Data...")
    return data_dict


data = generate_data(x_train,y_train,x_test,y_test)

class Game_Master():
    
    def __init__(self,target_size,vocab_size,bias,test_data,seed,episodes=1000):
        self.data = test_data
        self.vocab_size = vocab_size
        self.target_size = target_size
        self.speaker_bias = 1.0 - bias
        self.listener_bias = bias
        self.speaker = Speaker(512,self.target_size,self.vocab_size,self.speaker_bias,seed)
        self.listener = Listener(512,self.vocab_size,self.target_size,self.listener_bias,seed)
        self.episode_rewards = []
        self.accuracy_history = deque(maxlen=100)
        self.symbols_used = {}
        self.synonym_array = np.zeros((target_size,vocab_size))
        self.accuracy_record = []
        self.episodes = 1000

    def create_labels(self,num_labels,num_repeats):
        arr = np.array([])
        for i in range(num_labels):
            n = np.repeat(i,num_repeats)
            arr = np.concatenate((arr,n))
        np.random.shuffle(arr)
        return arr    

    def sample(self,label,test_or_train="X_train"):
        train = self.data[test_or_train]
        size = train[0].shape[0]
        target_one_hot = to_categorical(label,num_classes=self.target_size)
        target_image = 0
        image_list = []
        for i in range(self.target_size):
            sampled_image = np.expand_dims(train[i][np.random.choice(size),:],0)
            if i == label:
                target_image = sampled_image
            image_list.append(sampled_image)
        return target_image, image_list, target_one_hot

    def cross_entropy(self,predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
        return ce

    def test(self):
        y = self.create_labels(10,1)
        one_hot = to_categorical(y)
        it = 0
        while it < y.shape[0]:
            label = y[it]
            target_image, image_list, target_one_hot = self.sample(label,"X_test")
            speaker_action, speaker_probs = self.speaker.act(target_image)
            speaker_action_class = np.expand_dims(to_categorical(speaker_action,num_classes=self.vocab_size),0)
            image_list.append(speaker_action_class)
            listener_action, listener_probs = self.listener.act(image_list)
            if listener_action == label:
                speaker_reward = 1.0
                listener_reward = 1.0
            else:
                speaker_reward =  0.0
                listener_reward = 0.0
            self.synonym_array[int(label)][int(speaker_action)] += 1
            it += 1

    def play(self):
        y = self.create_labels(10,100)
        one_hot = to_categorical(y)
        bar = tqdm(np.arange(self.episodes))
        for i in bar:
            it = 0
            acc = np.zeros(self.target_size)
            right = 0
            while it < y.shape[0]:
                label = y[it]
                target_image, image_list, target_one_hot = self.sample(label)
                speaker_action, speaker_probs = self.speaker.act(target_image)
                speaker_action_class = np.expand_dims(to_categorical(speaker_action,num_classes=self.vocab_size),0)
                image_list.append(speaker_action_class)
                listener_action, listener_probs = self.listener.act(image_list)
                if listener_action == label:
                    speaker_reward = 1.0
                    listener_reward = 1.0
                    right += 1
                else:
                    speaker_reward =  0.0
                    listener_reward = 0.0
                acc = np.vstack((acc,listener_probs))
                self.speaker.save(target_image,speaker_action,speaker_probs,speaker_reward)
                self.listener.save(image_list,speaker_action_class,listener_action,listener_probs,listener_reward)
                it += 1
            accuracy = right/y.shape[0]
            total = np.sum(self.speaker.params["rewards"])
            cross = self.cross_entropy(acc[1:,:],one_hot)
            self.episode_rewards.append(total)
            self.accuracy_history.append(accuracy)
            self.accuracy_record.append(accuracy)
            bar.set_description("Cross Entropy: " + str(cross) + ", Rolling Acc 100: " + str(np.mean(self.accuracy_history)) + ", Accuracy: " + str(accuracy))
            self.speaker.train()     
            self.listener.train()
        self.test()


def test_without_bias():
    print("Test Without Bias")
    symbol_dict = {2:[],10:[],50:[],100:[],250:[],500:[],650:[],800:[],1000:[]}
    symbol_array = list(symbol_dict.keys())
    synonym_dict = {}
    accuracy_dict = {}
    iterations = 1

    for i in range(iterations):
        for symbol_num in symbol_array:
            print("Testing Game with vocab size: " + str(symbol_num) + " at iteration: " + str(0)) 
            gm = Game_Master(10,symbol_num,0.5,data,0)
            gm.play()
            symbol_dict[symbol_num].append(np.mean(gm.accuracy_history))
            synonym_dict[symbol_num] = gm.synonym_array
            accuracy_dict[symbol_num] = np.array(gm.accuracy_record)
        for key in symbol_dict.keys():
            print("Avg for vocab size for iteration " + str(i) + " : " + str(key) + ": " + str(np.mean(symbol_dict[key])))

    with h5py.File('datasets_csvs/synonym_without_bias.h5','w') as hf:
        for symbol in symbol_array:
            hf.create_dataset("synonym_arr_without_bias" + str(symbol),data=synonym_dict[symbol])

    gm.speaker.model.save("saved_models/speaker_without_bias.h5")
    gm.listener.model.save("saved_models/listener_without_bias.h5")

    df = pd.DataFrame.from_dict(accuracy_dict)
    df.to_csv("datasets_csvs/accuracy_results_1k_without_bias.csv", sep='\t')

def test_with_bias():
    print("Test With Bias")
    symbol_dict = {0.002:[],0.01:[],0.05:[],0.1:[],0.25:[],0.5:[],0.65:[],0.8:[],0.99:[]}
    bias_arr = list(symbol_dict.keys())
    synonym_dict = {}
    accuracy_dict = {}
    iterations = 1

    for i in range(iterations):
        for bias in bias_arr:
            print("Testing Game with Bias: " + str(bias) + " at iteration: " + str(i)) 
            gm = Game_Master(10,10,bias,data,0)
            gm.play()
            symbol_dict[bias].append(np.mean(gm.accuracy_history))
            synonym_dict[bias] = gm.synonym_array
            accuracy_dict[bias] = np.array(gm.accuracy_record)
        for key in symbol_dict.keys():
            print("Avg for bias for iteration " + str(i) + " : " + str(key) + ": " + str(np.mean(symbol_dict[key])))

    with h5py.File('datasets_csvs/synonym_bias.h5','w') as hf:
        for bias in bias_arr:
            hf.create_dataset("synonym_arr_with_bias" + str(bias),data=synonym_dict[bias])

    gm.speaker.model.save("saved_models/speaker_with_bias.h5")
    gm.listener.model.save("saved_models/listener_with_bias.h5")

    df = pd.DataFrame.from_dict(accuracy_dict)
    df.to_csv("datasets_csvs/accuracy_results_1k_with_bias.csv", sep='\t')

test_with_bias()
test_without_bias()

