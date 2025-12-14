
import time
from tensorflow import keras
from tensorflow.keras import layers

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


def run_mlp_gp_tuned(X_train_scaled, y_train, X_test_scaled, y_test):
    search_space = [
        Integer(16, 512, name="units"),
        Real(1e-4, 1e-1, prior="log-uniform", name="lr"),
        Real(0.0, 0.5, name="dropout"),
        Categorical(["adam", "sgd"], name="optimizer"),
    ]

    @use_named_args(search_space)
    def objective(units, lr, dropout, optimizer):
        model = keras.Sequential([
            layers.Dense(units, activation="relu",
                         input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ])

        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=lr)
        else:
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(
            X_train_scaled, y_train,
            epochs=30,
            batch_size=128,
            validation_data=(X_test_scaled, y_test),
            verbose=0
        )

        val_accuracy = history.history["val_accuracy"][-1]
        return -val_accuracy  # gp_minimize minimizes [web:50]

    print("\nRunning Bayesian Optimization...\n")
    start_time = time.time()

    bo_result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        random_state=42
    )

    bo_time = time.time() - start_time

    best_units, best_lr, best_dropout, best_optimizer = bo_result.x

    print("Best Hyperparameters Found:")
    print(f"Units        : {best_units}")
    print(f"Learning Rate: {best_lr:.6f}")
    print(f"Dropout      : {best_dropout}")
    print(f"Optimizer    : {best_optimizer}")

    model = keras.Sequential([
        layers.Dense(best_units, activation="relu",
                     input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(best_dropout),
        layers.Dense(1, activation="sigmoid"),
    ])

    if best_optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=best_lr)
    else:
        opt = keras.optimizers.SGD(learning_rate=best_lr, momentum=0.9)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    print("\nFINAL RESULTS — ONE-LAYER MLP + BAYESIAN OPTIMIZATION")
    print(f"Training Accuracy   : {train_acc:.4f}")
    print(f"Validation Accuracy : {val_acc:.4f}")
    print(f"Optimization Time   : {bo_time:.2f} seconds")

    return model, train_acc, val_acc, bo_time, bo_result
