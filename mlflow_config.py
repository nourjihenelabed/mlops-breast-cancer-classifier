"""
MLflow configuration and utility functions
"""
import mlflow
import os

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT_NAME = "breast-cancer-classification"

def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"✓ Experiment: {EXPERIMENT_NAME}")


def get_best_run(experiment_name=EXPERIMENT_NAME, metric="best_val_accuracy"):
    """Get the best run from an experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if runs.empty:
        print("No runs found")
        return None
    
    return runs.iloc[0]


def load_model_from_run(run_id):
    """Load a model from a specific run"""
    model_uri = f"runs:/{run_id}/model"
    return mlflow.keras.load_model(model_uri)


def compare_runs(experiment_name=EXPERIMENT_NAME):
    """Compare all runs in an experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_val_accuracy DESC"]
    )
    
    if runs.empty:
        print("No runs found")
        return None
    
    # Select relevant columns
    columns = [
        "run_id",
        "params.model_type",
        "params.units",
        "params.lr",
        "params.dropout",
        "params.optimizer",
        "metrics.best_val_accuracy",
        "metrics.best_val_loss"
    ]
    
    available_columns = [col for col in columns if col in runs.columns]
    return runs[available_columns]