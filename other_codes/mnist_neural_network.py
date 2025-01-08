from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# model.compile(
#     optimizer="sgd",
#     # loss="mse", # mse good for regression not classification
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

model.fit(
    train_images, keras.utils.to_categorical(train_labels), epochs=5, batch_size=32
)

model.evaluate(test_images, keras.utils.to_categorical(test_labels))
