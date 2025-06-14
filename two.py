import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def cube(num):
    model_path="two.h5"
    if (os.path.exists(model_path)):
        print("loading_model")
        model = keras.models.load_model(model_path)

    else:
        xs = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
        ys = np.array([1,8,27,64,125,216,343,512,729,1000], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=64, input_shape=[1], activation='relu'),
                                  keras.layers.Dense(32, activation='relu'),
                                  keras.layers.Dense(units=1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mean_squared_error")
        model.fit(xs,ys, epochs = 1500)
        model.save("two.h5")
    
    return model.predict(np.array([num]))[0][0]

print(cube(11))
print(cube(12))
print(cube(13))