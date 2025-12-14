from tensorflow import keras
from tensorflow.keras import layers
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import mlflow


def run_bayesian_optimization(X_train, y_train, X_val, y_val, input_dim):
    """
    Run Bayesian Optimization with MLflow tracking for each trial
    """
    
    search_space = [
        Integer(128, 600, name="units"),
        Real(1e-4, 5e-2, prior="log-uniform", name="lr"),
        Real(0.0, 0.5, name="dropout"),
        Categorical(["adam", "sgd"], name="optimizer")
    ]

    iteration = [0]  # Counter for nested runs

    @use_named_args(search_space)
    def objective(units, lr, dropout, optimizer):
        iteration[0] += 1
        
        # Create nested run for each optimization trial
        with mlflow.start_run(nested=True, run_name=f"trial_{iteration[0]}"):
            # Log trial parameters
            mlflow.log_param("trial_number", iteration[0])
            mlflow.log_param("units", int(units))
            mlflow.log_param("lr", float(lr))
            mlflow.log_param("dropout", float(dropout))
            mlflow.log_param("optimizer", optimizer)
            
            # Build model
            model = keras.Sequential([
                layers.Dense(units, activation="relu", input_shape=(input_dim,)),
                layers.Dropout(dropout),
                layers.Dense(1, activation="sigmoid")
            ])

            opt = (
                keras.optimizers.Adam(lr)
                if optimizer == "adam"
                else keras.optimizers.SGD(lr, momentum=0.9)
            )

            model.compile(
                optimizer=opt,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=128,
                validation_data=(X_val, y_val),
                verbose=0
            )

            # Get best validation accuracy
            best_val_acc = max(history.history["val_accuracy"])
            final_val_loss = history.history["val_loss"][-1]
            
            # Log metrics
            mlflow.log_metric("val_accuracy", best_val_acc)
            mlflow.log_metric("val_loss", final_val_loss)
            mlflow.log_metric("final_train_accuracy", history.history["accuracy"][-1])
            
            print(f"Trial {iteration[0]}: units={units}, lr={lr:.6f}, "
                  f"dropout={dropout:.2f}, opt={optimizer} -> val_acc={best_val_acc:.4f}")

            # Return negative accuracy (we want to maximize, but gp_minimize minimizes)
            return -best_val_acc

    print("\n🔍 Starting Bayesian Optimization (20 trials)...\n")
    result = gp_minimize(objective, search_space, n_calls=20, random_state=42)
    
    print(f"\n✅ Optimization complete!")
    print(f"Best validation accuracy: {-result.fun:.4f}")
    
    return result