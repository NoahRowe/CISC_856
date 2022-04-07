import tensorflow as tf
from tensorflow.keras import layers

########################################################################
############################ AGENT FUNCTIONS ###########################
########################################################################

def RNN_agent(observation_space, action_space):
    
    # Convert input to a ragged tensor
    observation_space_tensor = convet_to_ragged_tensor(observation_space)
    
    # Get maximum sequence length
    max_seq = observation_space_tensor.bounding_shape()[-1]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(len(action_space), activation='linear')
    ])
    
    # Can also use Huber loss?
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(5e-4),
                  metrics=['accuracy'])
    return model


def DNN_agent(observation_space, action_space):

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(observation_space),)),
        tf.keras.layers.Dense(len(action_space), activation='linear')
    ])
    
    # Can also use Huber loss?
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(5e-4),
                  metrics=['accuracy'])
    return model


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
    
def convert_to_1d(obs, single=True):
    if single:
        return [[x[0] if isinstance(x, list) else x for x in obs]]
    else:
        new_x = []
        for row in obs:
            new_x.append([x[0] if isinstance(x, list) else x for x in row])

        return new_x







