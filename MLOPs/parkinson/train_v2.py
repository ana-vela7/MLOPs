import pandas as pd
import sys
import joblib
import mlflow
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# # Set MinIO credentials and endpoint
# os.environ["AWS_ACCESS_KEY_ID"] = "F8MWOz0f0SEIyvKa66aY"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "BSPKB9TZ0vlI6dn8l5LPOa2hfWQRKttQg9YMXsL6"


def eval_metrics(actual, pred):
    """Evaluate metrics for the model."""
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, r2


def load_params():
    """Load parameters from params.yaml."""
    params_path = os.getenv("PARAMS_FILEPATH", "params.yaml")
    try:
        with open(f'{params_path}params.yaml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        return cfg
    except FileNotFoundError:
        print("Error: params.yaml file not found. Ensure it exists in the working directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing params.yaml: {e}")
        sys.exit(1)


def train_model(X_train_path, X_test_path, y_train_path, y_test_path, model_type, params):
    """Train a machine learning model and log with MLflow."""
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Get model parameters
    model_params = params['models'].get(model_type, {})
    if not model_params:
        print(f"Error: Model parameters for {model_type} not found in params.yaml.")
        sys.exit(1)

    # Initialize and train the model
    if model_type == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(**model_params)
    elif model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**model_params)
    else:
        print(f"Error: Unsupported model type {model_type}.")
        sys.exit(1)

    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    mae, r2 = eval_metrics(y_test, y_pred)

    print(f"{model_type} MAE: {mae}")
    print(f"{model_type} R2: {r2}")

    # Configure MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")  # Default to localhost if not set
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("MLflow Tracking URI:", tracking_uri)
    print("Experiment Name:", experiment_name)

    # Log model and metrics to MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, f"{model_type}_model")
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    return model


if __name__ == "__main__":
    # Command-line arguments
    if len(sys.argv) != 6:
        print("Usage: python train.py <X_train_path> <X_test_path> <y_train_path> <y_test_path> <model_type>")
        sys.exit(1)

    X_train_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_train_path = sys.argv[3]
    y_test_path = sys.argv[4]
    model_output = sys.argv[5]

    # Load parameters
    params = load_params()
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_output}_model.pkl"

    # Train and save the model
    try:
        model = train_model(X_train_path, X_test_path, y_train_path, y_test_path, model_output, params)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error during training or saving the model: {e}")
        sys.exit(1)
