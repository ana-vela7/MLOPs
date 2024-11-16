import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow
import yaml
import sys
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class DataExplorer:
    @staticmethod
    def explore_data(data):
        print("First 5 rows of the dataset:")
        print(data.head())
        print("\nStatistical Summary:")
        print(data.describe())
        print("\nData Information:")
        print(data.info())

    @staticmethod
    def plot_correlation_matrix(data):
        plt.figure(figsize=(12, 10))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    @staticmethod
    def plot_pca_variance(pca):
        varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Number of Components')
        plt.grid(True)
        plt.show()

class ParkinsonsModel:
    def __init__(self, filepath, model_type='GradientBoostingRegressor'):
        # Set MinIO credentials and endpoint
        os.environ["AWS_ACCESS_KEY_ID"] = "F8MWOz0f0SEIyvKa66aY"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "BSPKB9TZ0vlI6dn8l5LPOa2hfWQRKttQg9YMXsL6"
        self.filepath = filepath
        self.model_type = model_type
        if model_type == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif model_type == 'RandomForestRegressor':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.data = None
        self.X = None
        self.y = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        self.data.dropna(inplace=True)
        DataExplorer.explore_data(self.data)
        return self

    def preprocess_data(self):
        # Remove highly correlated columns
        correlated_cols = [
            'total_UPDRS', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'Shimmer:APQ11', 'Shimmer:DDA'
        ]
        self.y = self.data['total_UPDRS']
        self.data.drop(columns=['motor_UPDRS', 'total_UPDRS'] + correlated_cols, inplace=True)
        self.X = self.data.select_dtypes(include=['float64', 'int64'])

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Standardize the data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self

    def pca_analysis(self):
        pca = PCA()
        X_scaled = self.scaler.fit_transform(self.X)
        pca.fit(X_scaled)
        DataExplorer.plot_pca_variance(pca)
        varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
        num_componentes_90 = np.where(varianza_acumulada >= 0.9)[0][0] + 1
        print(f"Number of components to explain 90% variance: {num_componentes_90}")
        return self

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'\nMean Squared Error: {mse}')
        print(f'R² Score: {r2}')
        return mse, r2
    
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

def main():
    # Configure MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")  # Default to localhost if not set
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params_path = os.getenv("PARAMS_FILEPATH")
    filepath = f"{params_path}parkinsons_updrs.data"
    models = ['GradientBoostingRegressor', 'RandomForestRegressor']

    for model_type in models:
        model = ParkinsonsModel(filepath, model_type)
        (
            model.load_data()
                 .preprocess_data()
                 .pca_analysis()
                 .train_model()
        )
        mse, r2 = model.evaluate_model()

        # Log model and metrics to MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model.model, f"{model_type}_model")
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

        # Load parameters
        params = ParkinsonsModel.load_params()
        model_dir = params['data']['models']
        model_path = f"{model_dir}/{model_type}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")            

if __name__ == '__main__':
    main()
