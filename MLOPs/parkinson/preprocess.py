import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    
    # Remove highly correlated columns
    correlated_cols = [
        'total_UPDRS', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'Shimmer:APQ11', 'Shimmer:DDA'
    ]
    y = data['total_UPDRS']
    data.drop(columns=['motor_UPDRS', 'total_UPDRS'] + correlated_cols, inplace=True)
    X = data.select_dtypes(include=['float64', 'int64'])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)
