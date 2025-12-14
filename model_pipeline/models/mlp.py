from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(
    input_dim,
    units=500,
    dropout=0.3,
    optimizer="sgd",
    lr=0.01
):
    model = keras.Sequential([
        layers.Dense(units, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid")
    ])

    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
