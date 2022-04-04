#Testing Program

import gym
import numpy as np
from gym import spaces
import csv
import pandas as pd

import sys
sys.path.append("..")
from Environment import cryptoTrade
from tensorflow.keras import layers
import tensorflow as tf

def convet_to_ragged_tensor(obs, single=True):
    # Make sure nesting depth is consistent
    if single:
        for i, value in enumerate(obs):
            if not isinstance(value, list):
                obs[i] = list([value])

        return tf.ragged.constant([obs])

    else:
        for i, entry in enumerate(obs):
            for j, value in enumerate(entry):
                if not isinstance(value, list):
                    obs[i][j] = list([value])

        return tf.ragged.constant(obs)

data_path = "../data/Coinbase_BTCUSD_dailydata.csv"
env = cryptoTrade(data_path)
env.reset()

epsilon, max_epsilon, min_epsilon = 1, 1, 0.01
decay = 0.01

model = tf.keras.models.load_model('')

memory = []

X = []
y = []

for episode in range(300):
    total_training_rewards = 0
    
    observation = env.reset()
    done = False
    while not done:
        action = int(model.predict(convet_to_ragged_tensor(observation, single=True)).argmax())
            
        # Now step the simulation
        new_observation, reward, done = env.step(action)
        memory.append([observation, action, reward, new_observation, done])
            
        #observation = new_observation
        total_training_rewards += reward
        
        if done:
            print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
            #total_training_rewards += 1
            txt = "{:.2f}\n"
            

