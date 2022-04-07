#Stock trading environement
import itertools
import time
import numpy as np
import pandas as pd
import gym
from gym import spaces

from data.feature_creation import add_moving_mean, add_moving_max, add_moving_min


class cryptoTrade(gym.Env):


    def __init__(self, data_path, episode_size=60) -> None:
        super(cryptoTrade, self).__init__()
        
        self.init_bankroll = 1e6
        self.episode_size = episode_size
        
        self.n = 10 # Max buy or sell amount
        self.action_space = np.linspace(-self.n, self.n, num=2*self.n+1, dtype=int)
        
        self.lookback_period = 10 # Observation range
        
        # Define core features
        self.dim_2_features = ["open", "high", "low",  "close"]
        self.dim_1_features = ["num_shares", "buying_power", "net_worth", 
                               "avg_held_share_price", "held_shares_value", "first_bought_price"]
        
        self.raw_data = pd.read_pickle(data_path)
        self.preprocess_data()
        
        self.dim_1_observation_space = np.zeros(len(self.dim_1_features))
        self.dim_2_observation_space = np.zeros((len(self.dim_2_features), self.lookback_period))
        
        # Total observation space will be ragged tensor of dim_1 and dim_2 observations
        self.observation_space = [0 for _ in range(len(self.dim_1_observation_space)+len(self.dim_2_observation_space))]
        
        # Make sure the observations represent the first day
        self.reset()

        
    def preprocess_data(self):
        '''
        Currently only used to init the columns. preprocessing is done again after scalign 
        '''
        
        data = self.raw_data.copy()
        
        # Define the columns we want to add
        column_vals = self.dim_2_features
        window_vals = [5, 10, 15, 20] # In minutes
        
        for col, window in itertools.product(column_vals, window_vals):
            # Add moving average of all columns
            data, new_feature = add_moving_mean(data, col, window)
            self.dim_2_features.append(new_feature)
            # Add moving max of all columns
#             data, new_feature = add_moving_min(data, col, window)
#             self.dim_2_features.append(new_feature)
            # Add moving min of all columns
            data, new_feature = add_moving_max(data, col, window)
            self.dim_2_features.append(new_feature)
        
        # Remove missing data
        data.dropna(inplace=True)
        
        self.data = data

    
    def step(self, raw_action):
        
        # Update the pricing variables
        self.scaled_price = self.scaled_data['close'].iloc[self.current_index]
        self.real_price = self.scaled_price * self.scaling_values["close"]
        
        self.old_scaled_price = self.scaled_data['close'].iloc[self.current_index-1]
        self.old_real_price = self.old_scaled_price * self.scaling_values["close"]

        # Map the action to num of shares
        action = self.action_space[int(raw_action)]
        
        # Make sure it is a valid action
        valid_action = self.get_valid_action(action)
        
        # Buy/sell shares
        self.execute_action(valid_action)
        
        # Determine reward from this action
        reward = self.calculate_reward(valid_action)
        
        # Update net worth
        self.update_net_worth()
        
        # Shift observation forward one day
        self.current_index += 1
        self.update_array_observations()
        
        # Check if done
        done = self.is_done()
        
        return valid_action, self.observation_space, reward, done

    
    def get_valid_action(self, action):
        
        # Trying to buy shares
        if action>0:
            if self.real_price*action > self.buying_power: 
                return int(self.buying_power//self.real_price)

        elif action<0:
            if len(self.shares_held) < np.abs(action):
                return len(self.shares_held)

        return action

    
    def execute_action(self, action):
        # Have already checked that the given action is valid
        # Will also return the reward (sell_price-buy_price)

        self.buying_power -= self.real_price*action
        
        if action > 0:
            #### BUY ####
            for _ in range(action):
                self.shares_held.append(self.scaled_price)
            
        elif action < 0:
            #### SELL ####
            purchase_price = 0
            for _ in range(np.abs(action)):
                self.shares_held.pop(0)


    def calculate_reward(self, action):
        
        num_shares_kept = len(self.shares_held)
        
        # Profit from holding
        old_value = num_shares_kept*self.old_real_price
        new_value = num_shares_kept*self.real_price
        
        return new_value-old_value


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
        self.dim_1_observation_space[1] = self.buying_power/self.init_bankroll
        self.dim_1_observation_space[2] = self.net_worth/self.init_bankroll
        self.dim_1_observation_space[3] = np.mean(self.shares_held) if len(self.shares_held)>0 else 0
        self.dim_1_observation_space[4] = len(self.shares_held) * self.scaled_price
        self.dim_1_observation_space[5] = self.shares_held[0] if len(self.shares_held)>0 else 0


    def update_dim_2_observations(self):
        
        start_index, end_index = self.current_index-self.lookback_period, self.current_index
        for i, feature in enumerate(self.dim_2_features):
            self.dim_2_observation_space[i] = self.scaled_data[feature].iloc[start_index:end_index].values

            
    def scale_data(self):
        
        unscaled_columns = ['Date', "Volume"]
        scaled_data = self.data.copy()
        scaling_values = {}
        for column in scaled_data.columns:
            if column not in unscaled_columns:
                if scaled_data[column].iloc[self.current_index]!=0:
                    scaling_values[column] = scaled_data[column].iloc[self.current_index]
                    scaled_data[column] = scaled_data[column]/scaling_values[column]
                else:
                    scaling_values[column] = 1
    
        self.scaled_data = scaled_data
        self.scaling_values = scaling_values

    
    def reset(self):
        
        # Randomly choose somewhere in the data to iterate over for this episode
        self.current_index = np.random.randint(self.lookback_period, len(self.data)-self.episode_size)
        self.episode_end_index = self.current_index + self.episode_size
        
        # Scale the data
        self.scale_data()
        
        # Set the prices
        self.scaled_price = self.scaled_data['close'].iloc[self.current_index]
        self.real_price = self.scaled_price * self.scaling_values["close"]
        
        self.old_scaled_price = self.scaled_data['close'].iloc[self.current_index-1]
        self.old_real_price = self.scaled_price * self.scaling_values["close"]
        
        self.buying_power = self.init_bankroll
        self.net_worth = self.buying_power
        self.shares_held = []
        self.total_profit = 0
        
        self.update_array_observations()

        return self.observation_space
        

    def get_observation_states(self):
        return self.observation_space
        
    def update_net_worth(self):
        self.net_worth = self.buying_power + self.real_price*len(self.shares_held)
    
    def is_done(self):
        return self.current_index >= self.episode_end_index
    