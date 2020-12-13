import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# %%


seed = 42
# %%
tf.random.set_seed(seed)
model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(4))
learning_rate = ExponentialDecay(1e-2, 10000, 0.96)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
# %%
model.build()
model.summary()
