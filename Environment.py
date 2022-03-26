#Stock trading environement

import gym
import numpy as np
from gym import spaces
import csv


class cryptoTrade(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self) -> None:
        super(cryptoTrade,self).__init__()
        self.n = 10
        self.k = 10
        self.action_space = spaces.Discrete(2*self.n+1)
        self.action_space = np.linspace(-self.n,self.n,num=2*self.n+1)
        self.observation_space = spaces.Box(low=0,high=np.inf,shape=(self.k,7))
        self.currentday = self.k
        self.CurrentBalance = 1000
        self.CurrentShares = 0
        with open('data/Coinbase_BTCUSD_dailydata.csv') as csv_file:
            self.csv_reader = csv.reader(csv_file, delimiter=',')
    
    def step(self, action):
        done = False
        self.CurrentShares += action
        self.currentday += 1
        #Calculate Reward
        reward = self.CurrentShares*(self.csv_reader[self.currentday][4] - self.observation_space[self.k-1][4])
        #Adjust current balance based on action 
        if action > 0: #buy
            self.CurrentBalance -= action*self.observation_space[self.k-1][4]
        else: #sell 
            self.CurrentBalance += action*self.observation_space[self.k-1][4]
        count = 0
        for i in range(self.currentday-self.k,self.currentday+self.k):
            for j in  range(7):
                if j == 6:
                    self.observation_space[count][j] = self.csv_reader[i][j+1]
                else:
                    self.observation_space[count][j] = self.csv_reader[i][j]
            count += 1
        return self.observation_space,reward,done
    
    def render(self):
        return 
    
    def valid(self,action):
        if action*self.observation_space[self.k-1][4]>self.CurrentBalance:
            return False
        else:
            return True
    
    def reset(self):
        self.CurrentBalance = 1000
        self.currentday = self.k
        self.CurrentShares = 0
        
        for i in range(self.k):
            for j in range(7):
                if j==6:
                    self.observation_space[i][j] = self.csv_reader[i+1][j+1]
                else:
                    self.observation_space[i][j] = self.csv_reader[i+1][j]
        return self.observation_space