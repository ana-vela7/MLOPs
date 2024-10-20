from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
from sklearn.compose import ColumnTransformer
import mlflow
import pandas as pd

import logging


def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    rms = mean_squared_error(actual,pred)
    return mae, r2, rms

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Eliminar filas con valores nulos
df = pd.read_csv("parkinsons_updrs.data")
df = df.dropna()

# Preprocesamiento: eliminación de filas con valores faltantes y selección de columnas numéricas
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['motor_UPDRS', 'total_UPDRS', 'sex'])
y = df['total_UPDRS']

# Lista de columnas altamente correlacionadas
correlated_cols = ['Jitter:RAP','Jitter:PPQ5', 'Jitter:DDP', 'Shimmer(dB)', 'Shimmer:APQ3','Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA']
vars = X.drop(columns=['Jitter:RAP','Jitter:PPQ5', 'Jitter:DDP', 'Shimmer(dB)', 'Shimmer:APQ3','Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA']).columns.tolist()

svr_pol = svm.SVR(kernel='poly',C=100,gamma='auto',degree=3,epsilon=0.1,coef0=1)
svr_rbf = svm.SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = svm.SVR(kernel="linear", C=100, gamma="auto")

models = [svr_pol,svr_lin,svr_rbf]

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"/user/Parkinson")

for model in models:
    mlflow.start_run()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un ColumnTransformer para eliminar las columnas correlacionadas
    preprocessing = ColumnTransformer(
        transformers=[
            ('drop_correlated_cols', 'drop', correlated_cols),
            ('scaling',StandardScaler(),vars)
        ],
        remainder='passthrough'  # Mantener el resto de las variables
    )

    X_train_trans = preprocessing.fit_transform(X_train)
    X_test_trans = preprocessing.transform(X_test)
    model.fit(X_train_trans,y_train)
    y_pred = model.predict(X_test_trans)
    (mae, r2, rms) = eval_metrics(y_test, y_pred)
    params = model.get_params()

    for key, val in params.items():
        mlflow.log_param(key, val)

    mlflow.sklearn.log_model(model,'model')
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rms", rms)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    print("  rms: %s" % rms)
    mlflow.end_run()


