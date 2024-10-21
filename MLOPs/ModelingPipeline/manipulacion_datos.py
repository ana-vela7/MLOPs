from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
import mlflow
import pandas as pd

import logging

def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, r2

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Eliminar filas con valores nulos
df = pd.read_csv("parkinsons_updrs.data")
df = df.dropna()

# Preprocesamiento: eliminación de filas con valores faltantes y selección de columnas numéricas
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['motor_UPDRS', 'total_UPDRS', 'sex'])
y = df['total_UPDRS']

'''
Columnas altamente correlacionadas (con un umbral de correlación mayor a 0.9) y, por lo tanto, 
son candidatas para ser eliminadas del dataset:
total_UPDRS
Jitter:RAP
Jitter:PPQ5
Jitter:DDP
Shimmer(dB)
Shimmer:APQ3
Shimmer:APQ5
Shimmer:APQ11
Shimmer:DDA
'''

# Lista de columnas altamente correlacionadas
correlated_cols = []
for col in X:
    if 'total_UPDRS' in col or 'Jitter:RAP' in col or 'Jitter:PPQ5' in col or 'Jitter:DDP' in col or 'Shimmer(dB)' in col or 'Shimmer:APQ3' in col or 'Shimmer:APQ5' in col or 'Shimmer:APQ11' in col or 'Shimmer:DDA' in col:
        correlated_cols.append(col)

# Crear un ColumnTransformer para eliminar las columnas correlacionadas
preprocessing = ColumnTransformer(
    transformers=[
        ('drop_correlated_cols', 'drop', correlated_cols)
    ],
    remainder='passthrough'  # Mantener el resto de las variables
)

# Eliminar las columnas con alta correlación
df = df.drop(columns=correlated_cols)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"/user/Parkinson")

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with mlflow.start_run() as run:
    # Entrenar un modelo de Gradient Boosting Regressor
    gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr_model.fit(X_train, y_train)
    y_pred_gbr = gbr_model.predict(X_test)      
    mae_gbr, r2_gbr = eval_metrics(y_test, y_pred_gbr)
    print("Gradient Boosting Regressor MAE: %s" % mae_gbr)
    print("Gradient Boosting Regressor R2: %s" % r2_gbr)

    # Registrando el modelo en MLFLOW
    mlflow.sklearn.log_model(gbr_model, "gbr_model")
    print(f"Model saved to: runs:/{run.info.run_id}/model")        
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("r2", r2_gbr)
    mlflow.log_metric("mae", mae_gbr)    

with mlflow.start_run() as run:

    # Entrenar un modelo de Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mae_rf, r2_rf = eval_metrics(y_test, y_pred_rf)
    print("Random Forest Regressor MAE: %s" % mae_rf)
    print("Random Forest Regressor R2: %s" % r2_rf)    

    # Registrando el modelo en MLFLOW
    mlflow.sklearn.log_model(rf_model, "rf_model")
    print(f"Model saved to: runs:/{run.info.run_id}/model")       
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)  
    mlflow.log_metric("r2", r2_rf)
    mlflow.log_metric("mae", mae_rf)

with mlflow.start_run() as run:

    # Entrenar un modelo de Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mae_lr, r2_lr = eval_metrics(y_test, y_pred_lr)
    print("Linear Regression MAE: %s" % mae_lr)
    print("Linear Regression R2: %s" % r2_lr)

    # Registrando el modelo en MLFLOW
    mlflow.sklearn.log_model(lr_model, "lr_model")
    print(f"Model saved to: runs:/{run.info.run_id}/model")       
    # mlflow.log_param("n_estimators", 100)
    # mlflow.log_param("max_depth", 5)
    # mlflow.log_param("random_state", 42)  
    mlflow.log_metric("r2", r2_rf)
    mlflow.log_metric("mae", mae_rf)