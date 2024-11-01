# Documentación de Modelo de Predicción sobre la Enfermedad de Parkinson

## Descripción General del Modelo
Esta solución de Aprendizaje Automático está diseñada para predecir puntajes de UPDRS (Unified Parkinson's Disease Rating Scale) usando varias mediciones biomédicas. La implementación incluye varios modelos de regresión con su respectivo versionado de experimentos y prácticas de governanza de modelo.

## Procesamiento de Datos y Governanza

### Fuente de datos
- Dataset: "Parkinson's Disease UPDRS Dataset"
- Variable Objetivo: `total_UPDRS` (Score total de UPDRS)
- Features de Entrada: Mediciones biométricas
- Manejo de Datos: Implementa la eliminación de datos nulos mediante `dropna()`
- Protección de Datos: Aseguramiento de la no-existencia de datos personales identificables en el dataset.

### Feature Engineering
#### Control de calidad de datos
1. Manejo de datos nulos
   - Implementación de `dropna()` para asegurar que los datos estén completos
   - El manejo de valores faltantes disminuye el potencial de sesgo en el modelado

2. Análisis de Correlación
   - Se identificaron y removieron los atributos altamente correlacionados (correlation > 0.9):
   ```
   - Jitter:RAP
   - Jitter:PPQ5
   - Jitter:DDP
   - Shimmer(dB)
   - Shimmer:APQ3
   - Shimmer:APQ5
   - Shimmer:APQ11
   - Shimmer:DDA
   ```
   - Reduce problemas de colinearidad
   - Mejora la interpretabilidad del modelo

3. Preprocesamiento de datos
   - Implementación de `StandardScaler` para normalización de features
   - Separación train-test (80-20) con random_state para reproducibilidad

## Implementación de Modelo 

### Arquitectura
Tres modelos fueron implementados para su comparación:

1. Gradient Boosting Regressor
   - Parameters:
     - n_estimators: 100
     - learning_rate: 0.1
     - max_depth: 3
     - random_state: 42

2. Random Forest Regressor
   - Parameters:
     - n_estimators: 100
     - max_depth: 5
     - random_state: 42

3. Linear Regression
   - Default parameters

### Prácticas de governanza

#### 1. Tracking de experimentos
- Implemenatción de MLflow experiment tracking
- Tracking URI en local para reproducibilidad entre el equipo de trabajo (http://localhost:5000)
- Ruta dedicada de experimento (/user/Parkinson)

#### 2. Versionado
- Cada modelo es guardado de manera independiente mediante IDs únicos
- Identificadores por modelo:
  - "gbr_model"
  - "rf_model"
  - "lr_model"

#### 3. Registro de parámetros
- Hyperparameters de cada modelo
- Asegura la trazabilidad y reproducibilidad
- Facilita la comparación y optimización de los modelos

#### 4. Resgitro de Métricas
Dos Métricas por modelo:
- Mean Absolute Error (MAE)
- R-squared (R2) Score

#### 5. Implementación de logging
```python
mlflow.log_param("parameter_name", value)
mlflow.log_metric("metric_name", value)
mlflow.sklearn.log_model(model, "model_name")
```

## Medidas de calidad de código y Mejores Prácticas

### 1. Error Handling
- Se implementó logging con Python

### 2. Organización de Código
- Clara separación de reponsabilidades:
  - Data preprocessing
  - Model training
  - Evaluation
  - Experiment tracking

### 3. Reproducibilidad
- Random states fijos (42) para:
  - Separación Train-test
  - Inicialización de modelo
- Pipeline de preprocesamiento estándar para todos los modelos 

## Dependencias
- scikit-learn
- pandas
- mlflow
- logging

## Instrucciones de uso 
1. Asegurarse que el servidor de MLFLow esté corriendo en http://localhost:5000
2. El Archivo de Datos ("parkinsons_updrs.data") debe estar en el directorio de trabajo
3. Correr el script para entrenar y loggear los modelos

