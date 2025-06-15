import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def parab(num):
    model_path="six.keras"
    if (os.path.exists(model_path)):
        model = keras.models.load_model(model_path)

    else:
        xs = np.array(range(-20,21),dtype=float)
        ys = xs**2
        model = keras.Sequential([keras.layers.Dense(units=64, input_shape=[1], activation='relu'),
                                  keras.layers.Dense(32,activation='relu'),
                                  keras.layers.Dense(16,activation='relu'),
                                  keras.layers.Dense(units=1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(xs,ys,epochs=2000)
        model.save("six.keras")
    
    return model.predict(np.array([num]))[0][0]

print(parab(-22))
print(parab(21))
print(parab(23))