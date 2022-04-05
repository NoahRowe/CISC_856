import tensorflow as tf
from tensorflow.keras import layers

def RNN_agent(observation_space, action_space):
    
    init = tf.keras.initializers.he_uniform(seed=None)
    
    # Convert input to a ragged tensor
    observation_space_tensor = convet_to_ragged_tensor(observation_space)
    
    # Get maximum sequence length
    max_seq = observation_space_tensor.bounding_shape()[-1]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True),
        tf.keras.layers.LSTM(64, kernel_initializer=init),
        tf.keras.layers.Dense(len(action_space), activation='linear', kernel_initializer=init)
    ])
    
    # Can also use Huber loss?
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model


#### WILL ADD DNN AGENT HERE ####


########################################################################
########################### HELPER FUNCTIONS ###########################
########################################################################

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