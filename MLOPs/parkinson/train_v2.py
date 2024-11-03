import pandas as pd
import sys
import joblib
import mlflow
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, r2

def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

params = load_params()

def train_model(X_train_path, X_test_path, y_train_path, y_test_path, model_type):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test =  pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    model_params = params['models'][model_type]

    if model_type == 'GradientBoostingRegressor':
        # GradientBoostingRegressor model
        model = GradientBoostingRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred_gbr = model.predict(X_test)      
        mae_gbr, r2_gbr = eval_metrics(y_test, y_pred_gbr)
        print("Gradient Boosting Regressor MAE: %s" % mae_gbr)
        print("Gradient Boosting Regressor R2: %s" % r2_gbr)

    elif model_type == 'RandomForestRegressor':
        # RandomForestRegressor model
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred_gbr = model.predict(X_test)      
        mae_gbr, r2_gbr = eval_metrics(y_test, y_pred_gbr)
        print("Gradient Boosting Regressor MAE: %s" % mae_gbr)
        print("Gradient Boosting Regressor R2: %s" % r2_gbr)        

    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"{model_type}_model")
        if model_type == 'GradientBoostingRegressor':
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 3)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("r2", r2_gbr)
            mlflow.log_metric("mae", mae_gbr)
        elif model_type == 'RandomForestRegressor':
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            mlflow.log_param("random_state", 42)  
            mlflow.log_metric("r2", r2_gbr)
            mlflow.log_metric("mae", mae_gbr)              

    return model

if __name__=="__main__":
    X_train_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_train_path = sys.argv[3]
    y_test_path = sys.argv[4]
    model_output = sys.argv[5]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_output}_model.pkl"    
    
    model = train_model(X_train_path, X_test_path, y_train_path, y_test_path, model_output)
    joblib.dump(model, model_path)
    