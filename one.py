import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def growth(days):

    model_path = "one.h5"
    if (os.path.exists(model_path)):
        print("Loading model...")
        model = keras.models.load_model(model_path)

    else:
        xs = np.array([1,2,3,4,5,6,7,8], dtype=float)
        ys = np.array([1,4,9,16,25,36,49,64], dtype=float)
        model = keras.Sequential([keras.layers.Dense(units=64, input_shape=[1], activation = 'relu'),
                                  keras.layers.Dense(64,activation = "relu"),
                                   keras.layers.Dense(units=1)
                                   ])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss="mean_squared_error")
        model.fit(xs,ys,epochs=2000)
        model.save("one.h5")
    
    return model.predict(np.array([days]))[0][0]

print(growth(9))
print(growth(10))
print(growth(11))