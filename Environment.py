#Stock trading environement

import gym
import numpy as np
from gym import spaces
import csv
import pandas as pd


class cryptoTrade(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, data_path) -> None:
        super(cryptoTrade,self).__init__()
        
        self.n = 10 # action range
        self.lookback_period = 10 # Observation range
        
        # Define features we will lookback on
        self.dim_2_features = ["unix", "low", "high", "open", "close", "volume", "vol_fiat"]
        self.dim_1_features = ["num_shares", "net_worth"]
        
        self.action_space = np.linspace(-self.n,self.n,num=2*self.n+1)
        
        self.dim_2_observation_space = np.zeros((len(self.dim_2_features), self.lookback_period))
        self.dim_1_observation_space = np.zeros(len(self.dim_1_features))
        
        # Total observation space will be ragged tensor of dim_1 and dim_2 observations
        self.observation_space = [0 for _ in range(len(self.dim_1_observation_space)+len(self.dim_2_observation_space))]
        
        self.currentday = self.lookback_period
        self.CurrentBalance = 1000
        self.CurrentShares = 0
        
        self.data = pd.read_csv(data_path)
        
    
    def step(self, raw_action):

        # Map the action to num of shares
        action = self.action_space[int(raw_action)]

        self.CurrentShares += action
        self.currentday += 1
        
        self.yesterday_price = self.data['close'].iloc[self.currentday-1]
        self.today_price = self.data['close'].iloc[self.currentday]
        
        # Calculate Reward (based on today's price)
        reward = self.today_price * self.CurrentShares
        
        # Adjust current balance based on action (bought with this action yesterday)
        self.CurrentBalance -= action * self.yesterday_price # Will account for selling w negative
        
        # Shift observation forward one day
        # self.update_array_observations()
        
        # Check if done
        done = self.is_done()
        
        return self.observation_space, reward, done
    
    def is_done(self):
        return self.currentday >= self.data.shape[0]-1
    
    def render(self):
        return 
    
    def valid(self,action):
        return action*self.observation_space[self.k-1][4] < self.CurrentBalance
    
    def update_array_observations(self):
        self.update_dim_1_observations()
        self.update_dim_2_observations()
        
        # Combine them together
        num_dim_1_features = len(self.dim_1_observation_space)
        num_dim_2_features = len(self.dim_2_observation_space)
        
        for i in range(num_dim_1_features):
            self.observation_space[i] = self.dim_1_observation_space[i]
        
        for i in range(num_dim_2_features):
            self.observation_space[i+num_dim_1_features] = self.dim_2_observation_space[i].tolist()
                
    def update_dim_1_observations(self):
        
        self.dim_1_observation_space[0] = self.CurrentShares
        self.dim_1_observation_space[1] = self.CurrentBalance
        
    def update_dim_2_observations(self):
        
        start_index, end_index = self.currentday-self.lookback_period, self.currentday
        for i, feature in enumerate(self.dim_2_features):
            self.dim_2_observation_space[i] = self.data[feature].iloc[start_index:end_index].values
    
    def reset(self):
        self.CurrentBalance = 1000
        self.currentday = self.lookback_period
        self.CurrentShares = 0
        
        self.update_array_observations()

        return self.observation_space