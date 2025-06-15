import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def multthree(num):
    model_path="five.h5"
    if(os.path.exists(model_path)):
        model = keras.models.load_model(model_path)
    else:
        xs = np.array([1, 2, 3, 4, 5], dtype=float)
        ys = np.array([3, 6.2, 9.5, 12.9, 16.4], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=128, input_shape=[1], activation='relu'),
                                  keras.layers.Dense(64, activation='relu'),
                                  keras.layers.Dense(32, activation='relu'),
                                  keras.layers.Dense(16, activation='relu'),
                                  keras.layers.Dense(8, activation='relu'),
                                  keras.layers.Dense(4, activation='relu'),
                                  keras.layers.Dense(2, activation='relu'),
                                  keras.layers.Dense(units=1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
        model.fit(xs,ys,epochs=1000)
        model.save('five.h5')
    
    return model.predict(np.array([num]))[0][0]

print(multthree(6))
print(multthree(7))
print(multthree(8))
        

