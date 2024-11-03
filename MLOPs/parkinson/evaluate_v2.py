import pandas as pd
import sys
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os


def evaluate_model(model_path, X_test_path, y_test_path, output_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}')

    write_evaluation_report(output_path, mae, r2)

def write_evaluation_report(file_path, mae, r2):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write("MAE:\n")
        f.write(str(mae))
        f.write("\nR2:\n")
        f.write(str(r2))         

if __name__=="__main__":
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    output_path = sys.argv[4]
    evaluate_model(model_path, X_test_path, y_test_path, output_path)