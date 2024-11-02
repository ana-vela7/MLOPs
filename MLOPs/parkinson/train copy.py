import pandas as pd
import sys
import joblib

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Flatten y_train to a 1D array
    y_train = y_train.values.ravel()
    # GradientBoostingRegressor model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__=="__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_output = sys.argv[3]
    
    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_output)
    