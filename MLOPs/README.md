# Parkinson

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Just for testging purposes

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Parkinson and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Parkinson   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Parkinson a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

Implementacion

Pre-Requisitos

- Instalar un Ambiente de Python
- Activarlo para terminal como para el Kernel
- Actualizar pip
- Instalar dependecias necesarias
- Instalar dvc
- Instalar mlflow 

1 - Crear un **MLOps template** usando cookiecutter

2 - Inicializar DVC
    * cd en el Folder de trabajo
    * git init
    * dvc init
    * Apuntar a un folder como local storage de DVC dvc remote add -d local_remote /Users/your_username/Documents/Demo1/local_storage

3 Agregando Archivos para versionado con DVC
   * dvc add FileName
   * git add FileName.dvc .gitignore
   * git commit -m "Comentario"
   * dvc push

4 Conectando DVC con mlflow
    * mlflow ui
    * Abrir mlflow en la siguientye direccion **http://127.0.0.1:5000**

5 Creacion de Archivos py (Parkinson Folder) y ipynb(Notebooks Folder) diferentes etapas
    * Los archivos contenidos en folder "Parkinson" son los que se usan para el pipeline dvc (dvc.yaml)
    * Los Archivos v1 ejecutan un solo modelo GradientBoostingRegressor
    * Los archivos v2 ejecutan dos modelo GradientBoostingRegressor y RandomForestRegressor, a su ves registran los los y metricas en mlflow
    * Los archivos Refactor v1 y v2 en el Folder Parkinson son el resultado de la refactorizacion de los notebooks y se pueden ejecutar de forma individual

6 Ejecutando la pipeline V1
    * renombrar el archivo dvc_v1.yaml y params_v1.yaml como dvc.yaml y param.yaml
    * modificar la ruta del archivo params.yaml con la ubicacion del dataset
    * Inicializar la pipeline -> dvc dag
    * Ejecutar la pipeline -> dvc repro

7 Ejecutar la pipeline v2
    * renombrar el archivo dvc_v2.yaml y params_v2.yaml como dvc.yaml y param.yaml
    * modificar la ruta del archivo params.yaml con la ubicacion del

8 Crear la imagen de Docker
    * docker build -t mlopsequipo12 .