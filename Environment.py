#Stock trading environement

import gym
import numpy as np
from gym import spaces
import csv
import pandas as pd


class cryptoTrade(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, data_path) -> None:
        super(cryptoTrade, self).__init__()
        
        self.n = 10 # action range
        self.lookback_period = 10 # Observation range
        self.init_bankroll = 1e6
        
        # Define features we will lookback on
        self.dim_2_features = ["low", "high", "open", "close", "volume", "vol_fiat"]
        self.dim_1_features = ["num_shares", "buying_power", "net_worth", 
                               "avg_held_share_price", "held_shares_value"]
        
        self.action_space = np.linspace(-self.n, self.n, num=2*self.n+1, dtype=int)
        
        self.dim_1_observation_space = np.zeros(len(self.dim_1_features))
        self.dim_2_observation_space = np.zeros((len(self.dim_2_features), self.lookback_period))
        
        # Total observation space will be ragged tensor of dim_1 and dim_2 observations
        self.observation_space = [0 for _ in range(len(self.dim_1_observation_space)+len(self.dim_2_observation_space))]
        
        self.current_day = self.lookback_period
        self.buying_power = self.init_bankroll
        self.net_worth = self.buying_power
        self.total_profit = 0
        self.shares_held = []
        
        self.data = pd.read_csv(data_path)
        
        self.today_price = self.data['close'].iloc[self.current_day]
        
        # Make sure the observations represent the first day
        self.reset()
        
    
    def step(self, raw_action):
        
        # Update the pricing variables
        self.today_price = self.data['close'].iloc[self.current_day]

        # Map the action to num of shares
        action = self.action_space[int(raw_action)]
        
        # Make sure it is a valid action
        valid_action = self.get_valid_action(action)
        
        # Buy/sell shares and calculate profit
        profit = self.execute_action(valid_action)
        self.total_profit += profit
        
        # Update net worth
        self.update_net_worth()
        
        # Shift observation forward one day
        self.current_day += 1
        self.update_array_observations()
        
        # Check if done
        done = self.is_done()
        
        return valid_action, self.observation_space, self.total_profit, done
    
    def get_valid_action(self, action):
        
        # Trying to buy shares
        if action>0:
            if self.today_price*action > self.buying_power: 
                # Can't afford all the shares, buy as many as possible
                return int(self.buying_power//self.today_price)
                
        # Trying to sell shares
        elif action<0:
            if len(self.shares_held) < np.abs(action):
                # Trying to sell too many shares, sell all of them
                return len(self.shares_held)

        return action

    
    def execute_action(self, action):
        # Have already checked that the given action is valid
        # Will also return the reward (sell_price-buy_price)

        self.buying_power -= self.today_price*action
        
        if action > 0:
            
            #### BUY ####
            for _ in range(action):
                self.shares_held.append(self.today_price)
            return 0
            
        elif action < 0:
            
            #### SELL ####
            purchase_price = 0
            for _ in range(np.abs(action)):
                purchase_price += self.shares_held.pop(0)
            return self.today_price*np.abs(action) - purchase_price
                
        else:
            #### NOTHING ####
            return 0
        
    def update_net_worth(self):
        self.net_worth = self.buying_power + self.today_price*len(self.shares_held)
    
    def is_done(self):
        return self.current_day >= self.data.shape[0]-1
    
    def render(self):
        return 
    
    def valid(self,action):
        return action*self.observation_space[self.k-1][4] < self.buying_power
    
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
        
        self.dim_1_observation_space[0] = len(self.shares_held)
        self.dim_1_observation_space[1] = self.buying_power
        self.dim_1_observation_space[2] = self.net_worth
        self.dim_1_observation_space[3] = np.mean(self.shares_held) if len(self.shares_held)>0 else 0
        self.dim_1_observation_space[4] = len(self.shares_held) * self.today_price
        
    def update_dim_2_observations(self):
        
        start_index, end_index = self.current_day-self.lookback_period, self.current_day
        for i, feature in enumerate(self.dim_2_features):
            self.dim_2_observation_space[i] = self.data[feature].iloc[start_index:end_index].values
    
    def reset(self):
        self.buying_power = self.init_bankroll
        self.net_worth = self.buying_power
        self.current_day = self.lookback_period
        self.shares_held = []
        self.total_profit = 0
        
        self.update_array_observations()

        return self.observation_space
    
    def get_observation_states(self):
        return self.observation_space
    
    