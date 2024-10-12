import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
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
        return self

def main():
    filepath = "parkinsons_updrs.data"
    model = ParkinsonsModel(filepath)
    (
        model.load_data()
             .preprocess_data()
             .pca_analysis()
             .train_model()
             .evaluate_model()
    )

if __name__ == '__main__':
    main()
