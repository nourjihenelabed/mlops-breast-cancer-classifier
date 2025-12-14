from model_pipeline.data_loading import load_data
from model_pipeline.preprocessing import preprocess
from model_pipeline.models.mlp import build_mlp
from model_pipeline.tuned.mlp_gp import run_bayesian_optimization

import sys
import joblib
import mlflow
import mlflow.keras
from datetime import datetime

# ==========================
# MLFLOW SETUP
# ==========================
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("breast-cancer-classification")

# ==========================
# STATE
# ==========================
state = {
    "data_loaded": False,
    "preprocessed": False,
    "baseline_trained": False,
    "best_params": False,
    "final_model": False
}

df = None
X_train = X_val = X_test = None
y_train = y_val = y_test = None

baseline_model = None
best_model = None
bo_result = None


def print_status():
    print("\nSTATUS")
    print(f"Data loaded     : {state['data_loaded']}")
    print(f"Preprocessed    : {state['preprocessed']}")
    print(f"Baseline trained: {state['baseline_trained']}")
    print(f"Best params     : {state['best_params']}")
    print(f"Final model     : {state['final_model']}")


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    import numpy as np
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm


def main():
    global df, X_train, X_val, X_test
    global y_train, y_val, y_test
    global baseline_model, best_model, bo_result

    while True:
        print_status()
        print("""
1 - Data loading
2 - Data preprocessing
3 - Train baseline MLP
4 - Train hypertuned MLP
5 - Save baseline model
6 - Save hypertuned model
7 - Evaluate baseline model
8 - Evaluate hypertuned model
0 - Exit
""")

        choice = input("Choice: ").strip()

        # 1️⃣ DATA LOADING
        if choice == "1":
            df = load_data("data/raw/data.csv")
            state["data_loaded"] = True
            print("✓ Data loaded")
            print(f"  Shape: {df.shape}")

        # 2️⃣ PREPROCESSING
        elif choice == "2":
            if not state["data_loaded"]:
                print("✗ Load data first")
                continue

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                scaler,
                label_encoder
            ) = preprocess(df)

            # SAVE PREPROCESSING ARTIFACTS
            joblib.dump(scaler, "artifacts/models/scaler.joblib")
            joblib.dump(label_encoder, "artifacts/models/label_encoder.joblib")

            state["preprocessed"] = True
            print("✓ Data preprocessed")
            print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            print("✓ Scaler and label encoder saved")

        # 3️⃣ BASELINE
        elif choice == "3":
            if not state["preprocessed"]:
                print("✗ Preprocess data first")
                continue

            print("\n🚀 Training baseline MLP...")
            
            with mlflow.start_run(run_name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_type", "baseline_mlp")
                mlflow.log_param("units", 256)
                mlflow.log_param("dropout", 0.3)
                mlflow.log_param("optimizer", "adam")
                mlflow.log_param("lr", 0.001)
                mlflow.log_param("epochs", 100)
                mlflow.log_param("batch_size", 128)
                mlflow.log_param("input_dim", X_train.shape[1])
                
                # Build and train
                baseline_model = build_mlp(input_dim=X_train.shape[1])

                history = baseline_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data=(X_val, y_val),
                    verbose=1
                )

                # Log metrics
                best_val_acc = max(history.history['val_accuracy'])
                best_val_loss = min(history.history['val_loss'])
                final_train_acc = history.history['accuracy'][-1]
                
                mlflow.log_metric("best_val_accuracy", best_val_acc)
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                
                # Log model
                mlflow.keras.log_model(baseline_model, "model")
                
                # Log training history as artifact
                import json
                history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                with open("temp_history.json", "w") as f:
                    json.dump(history_dict, f, indent=2)
                mlflow.log_artifact("temp_history.json", "training_history")
                
                print(f"\n✓ Baseline best val acc: {best_val_acc:.4f}")
                print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")
                
            state["baseline_trained"] = True

        # 4️⃣ HYPERTUNED
        elif choice == "4":
            if not state["preprocessed"]:
                print("✗ Preprocess data first")
                continue

            print("\n🔍 Running Bayesian Optimization...")
            
            with mlflow.start_run(run_name=f"hypertuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Run optimization
                bo_result = run_bayesian_optimization(
                    X_train, y_train, X_val, y_val, X_train.shape[1]
                )

                units, lr, dropout, optimizer = bo_result.x
                
                print("\nBest hyperparameters:")
                print(f"Units     : {units}")
                print(f"LR        : {lr}")
                print(f"Dropout   : {dropout}")
                print(f"Optimizer : {optimizer}")

                # Log hyperparameters
                mlflow.log_param("model_type", "hypertuned_mlp")
                mlflow.log_param("units", int(units))
                mlflow.log_param("lr", float(lr))
                mlflow.log_param("dropout", float(dropout))
                mlflow.log_param("optimizer", optimizer)
                mlflow.log_param("epochs", 100)
                mlflow.log_param("batch_size", 128)
                mlflow.log_param("input_dim", X_train.shape[1])
                mlflow.log_param("optimization_calls", 20)
                mlflow.log_metric("best_bo_score", -bo_result.fun)

                print("\n🚀 Training final model with best params...")
                
                # Train final model
                best_model = build_mlp(
                    input_dim=X_train.shape[1],
                    units=int(units),
                    dropout=float(dropout),
                    optimizer=optimizer,
                    lr=float(lr)
                )

                history = best_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data=(X_val, y_val),
                    verbose=1
                )

                # Log metrics
                best_val_acc = max(history.history['val_accuracy'])
                best_val_loss = min(history.history['val_loss'])
                final_train_acc = history.history['accuracy'][-1]
                
                mlflow.log_metric("best_val_accuracy", best_val_acc)
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("final_train_accuracy", final_train_acc)
                
                # Log model
                mlflow.keras.log_model(best_model, "model")
                
                # Log training history
                import json
                history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                with open("temp_history.json", "w") as f:
                    json.dump(history_dict, f, indent=2)
                mlflow.log_artifact("temp_history.json", "training_history")

                print(f"\n✓ Tuned best val acc: {best_val_acc:.4f}")
                print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")
                
            state["best_params"] = True
            state["final_model"] = True

        # 5️⃣ SAVE BASELINE
        elif choice == "5":
            if not state["baseline_trained"]:
                print("✗ Train baseline first")
                continue

            baseline_model.save("artifacts/models/baseline_mlp")
            print("✓ Baseline saved to artifacts/models/baseline_mlp")

        # 6️⃣ SAVE TUNED
        elif choice == "6":
            if not state["final_model"]:
                print("✗ Train tuned model first")
                continue

            best_model.save("artifacts/models/hypertuned_mlp.keras")
            print("✓ Tuned model saved to artifacts/models/hypertuned_mlp.keras")

        # 7️⃣ EVALUATE BASELINE
        elif choice == "7":
            if not state["baseline_trained"]:
                print("✗ Train baseline first")
                continue
            
            print("\n📊 Evaluating baseline model on test set...")
            metrics, cm = evaluate_model(baseline_model, X_test, y_test)
            
            print("\nTest Set Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(cm)

        # 8️⃣ EVALUATE TUNED
        elif choice == "8":
            if not state["final_model"]:
                print("✗ Train tuned model first")
                continue
            
            print("\n📊 Evaluating hypertuned model on test set...")
            metrics, cm = evaluate_model(best_model, X_test, y_test)
            
            print("\nTest Set Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(cm)

        # 0️⃣ EXIT
        elif choice == "0":
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()