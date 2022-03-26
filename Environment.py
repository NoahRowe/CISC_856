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
        self.action_space = np.linspace(-self.n,self.n)
        self.observation_space = spaces.Box(low=0,high=np.inf,shape=(self.k,7))
        self.currentday = self.k
        self.CurrentBalance = 1000
        self.CurrentShares = 0
        with open('data/Coinbase_BTCUSD_dailydata.csv') as csv_file:
            self.csv_reader = csv.reader(csv_file, delimiter=',')
    
    def step(self, action):
        self.CurrentShares += action

        self.currentday += 1

        done = False
        reward = 0
        return self.observation_space,reward,done
    
    def render(self):
        return 
    
    def reset(self):
        self.CurrentBalance = 1000
        
        for i in range(self.k):
            for j in range(7):
                if j==6:
                    self.observation_space[i][j] = self.csv_reader[i+1][j+1]
                else:
                    self.observation_space[i][j] = self.csv_reader[i+1][j]
        return self.observation_space