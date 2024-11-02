import pandas as pd
import sys as sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(data):
    data = pd.read_csv(data)
    X = data.select_dtypes(include=['float64', 'int64']).drop(columns=['motor_UPDRS', 'total_UPDRS', 'sex'])
    y = data['total_UPDRS']

    # Define preprocessing
    # Removing High correlated columns
    correlated_cols = []
    for col in X:
        if 'total_UPDRS' in col or 'Jitter:RAP' in col or 'Jitter:PPQ5' in col or 'Jitter:DDP' in col or 'Shimmer(dB)' in col or 'Shimmer:APQ3' in col or 'Shimmer:APQ5' in col or 'Shimmer:APQ11' in col or 'Shimmer:DDA' in col:
            correlated_cols.append(col)

    # Define preprocessing
    preprocessing = ColumnTransformer(
        transformers=[
            ('drop_correlated_cols', 'drop', correlated_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.90))
    ])
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__=="__main__":
    data = sys.argv[1]
    output_train = sys.argv[2]
    output_test = sys.argv[3]
    output_target_train = sys.argv[4]
    output_test_target = sys.argv[5]
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    pd.DataFrame(X_train).to_csv(output_train, index=False)
    pd.DataFrame(X_test).to_csv(output_test, index=False)
    pd.DataFrame(y_train).to_csv(output_target_train, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)  