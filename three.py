import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def square_root(num):
    model_path="three.h5"
    if(os.path.exists(model_path)):
        print("loading model...")
        model = keras.models.load_model(model_path)

    else:
        xs =np.array([1,4,9,16,25,36,49,64,81,100], dtype=float)
        ys =np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
        model =keras.Sequential([keras.layers.Dense(units=128, input_shape=[1], activation = 'relu'),
                                 keras.layers.Dense(64, activation='relu'),
                                 keras.layers.Dense(32, activation='relu'),
                                 keras.layers.Dense(16, activation='relu'),
                                 keras.layers.Dense(units=1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mean_squared_error")
        model.fit(xs,ys, epochs=1000)
        model.save("three.h5")

    return model.predict(np.array([num]))[0][0]

print(square_root(121))
print(square_root(144))
print(square_root(169)) 
