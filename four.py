import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def power(num):
    model_path = "four.h5"
    if (os.path.exists(model_path)):
        model = keras.models.load_model(model_path)
    else:
        xs= np.array(range(1,16), dtype=float)
        ys= np.log2(2**xs)
        model= keras.Sequential([keras.layers.Dense(units=128, input_shape=[1], activation='relu'),
                                 keras.layers.Dense(64, activation='relu'),
                                 keras.layers.Dense(32, activation='relu'), 
                                 keras.layers.Dense(16, activation='relu'),
                                 keras.layers.Dense(8, activation='relu'),
                                 keras.layers.Dense(units=1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(xs,ys,epochs=2000)
        model.save("four.h5")

    return 2**model.predict(np.array([num]))[0][0]

print(power(11))
print(power(12))
print(power(13))
print(power(14))  # Should be ~16384
print(power(15))  # ~32768
print(power(16))  # ~65536
