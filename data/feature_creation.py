'''
File that contains all feature preprocessing done when the data
is originally loaded into the environemnt. 
'''

def add_moving_mean(data, column, window):
    # Rolling function ensures no data leakage
    feature_name = f"{column}_mean_{window}"
    data[feature_name] = data[column].rolling(window=window).mean()
    return data, feature_name

def add_moving_max(data, column, window):
    feature_name = f"{column}_max_{window}"
    data[feature_name] = data[column].rolling(window=window).max()
    return data, feature_name

def add_moving_min(data, column, window):
    feature_name = f"{column}_min_{window}"
    data[feature_name] = data[column].rolling(window=window).min()
    return data, feature_name