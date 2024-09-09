from keras import Sequential
from keras import layers
from keras import optimizers

def trainModel(x, y):
    model = Sequential()
    model.add(layers.Dense(1, input_shape=(1,), activation="linear"))

    optimizador = optimizers.SGD(learning_rate=0.00001)

    model.compile(loss="mse", optimizer=optimizador)

    batch_size = len(x)
    history = model.fit(x, y, epochs=10000, batch_size=batch_size)

    return model, history
