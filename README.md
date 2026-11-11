# Predicción de precios de casas (House Prices)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de la Clase 2 de Product Development (Maestría, Cuarto Trimestre). Predice el precio de venta (`SalePrice`) de casas en Ames, Iowa, a partir del dataset de la competencia [House Prices de Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

El flujo completo va de los datos crudos a un pipeline de inferencia que recibe datos nuevos y devuelve el precio en dólares, cuidando que no haya fuga de datos entre entrenamiento y validación.

## Resultados

Torneo de 6 modelos optimizados con `GridSearchCV` y validación cruzada de 10 folds, comparados en un set de validación (20% de `train.csv`):

| Modelo | RMSE real (validación) | NRMSE al rango |
|---|---|---|
| **Gradient Boosting** (ganador) | **$27,615** | **0.0384** |
| Ridge | $28,596 | 0.0397 |
| Random Forest | $28,649 | 0.0398 |
| ElasticNet | $28,753 | 0.0400 |
| Lasso | $28,758 | 0.0400 |
| MLPRegressor | $29,660 | 0.0412 |

El modelo base (predecir siempre la mediana) tiene un RMSE de $88,667. Los 6 modelos generan predicciones coherentes en test.

## Requisitos

- [Anaconda o Miniconda](https://www.anaconda.com/download)
- Python 3.11 (lo instala el entorno de conda)

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/25005389-art/Proyecto1.git
   cd Proyecto1
   ```

2. Crear y activar el entorno `env_house_pricing_a` (instala las dependencias de `requirements.txt` y el módulo `clase_2`):

   ```bash
   conda env create -f environment.yml
   conda activate env_house_pricing_a
   ```

3. Descargar los datos de la [competencia en Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) y colocarlos en `data/raw/`. Los datos no se versionan en el repositorio:

   ```
   data/raw/
   ├── train.csv
   ├── test.csv
   ├── data_description.txt
   └── sample_submission.csv
   ```

## Flujo de trabajo

Los notebooks se ejecutan en orden desde la carpeta `notebooks/`. Cada uno lee y actualiza `pipeline_config.yaml`, que guarda las decisiones y rutas del flujo.

| Notebook | Qué hace | Genera |
|---|---|---|
| `1.0-bz-analisis-exploratorio` | Análisis exploratorio y selección de las 30 features (top 10 de continuas, discretas y categóricas) usando solo el 80% de entrenamiento | `MODEL_FEATURES` |
| `2.0-bz-preprocesamiento-modelo` | Pipeline de feature engineering con feature-engine y división train / validación 80/20 | `models/preprocesamiento_pipeline.joblib`, `data/processed/` |
| `3.0-bz-entrenamiento-modelos` | Torneo de 6 modelos con validación cruzada, comparación en validación y reentrenamiento del ganador con el 100% de train | `models/modelo_ganador.joblib`, `reports/torneo_*` |
| `4.0-bz-inferencia` | Predicciones sobre los datos crudos de test y pipeline de inferencia en dólares | `models/pipeline_inferencia.joblib`, `data/processed/predicciones_test.csv` |
| `5.0-bz-evaluacion-modelos` | Evaluación de la coherencia de las salidas de los 6 modelos | `reports/evaluacion_modelos.json` |

### Feature engineering

El pipeline de preprocesamiento aplica, en orden:

1. Imputación: moda en categóricas y mediana en numéricas.
2. Codificación: agrupación de categorías raras (< 1%) y codificación ordinal ordenada por el target; las categorías no vistas se codifican como -1.
3. Tratamiento de outliers: *capping* por rango intercuartil en las variables continuas.
4. Transformación Yeo-Johnson de las continuas con |sesgo| > 0.75.
5. Escalado con `StandardScaler`.

El target se modela como `log(1 + SalePrice)`.

### Prevención de fuga de datos

- La selección de features usa solo el 80% de entrenamiento.
- En la validación cruzada, el pipeline de feature engineering se reajusta dentro de cada fold.
- El set de validación no participa en ningún ajuste; solo se usa para comparar modelos.

## Inferencia

El pipeline de inferencia recibe datos crudos con las columnas de `MODEL_FEATURES` y devuelve `SalePrice` en dólares:

```python
import joblib
import pandas as pd
import yaml

with open("pipeline_config.yaml", encoding="utf-8") as archivo:
    config = yaml.safe_load(archivo)

pipeline_inferencia = joblib.load("models/pipeline_inferencia.joblib")
datos = pd.read_csv("data/raw/test.csv", index_col="Id")
precios = pipeline_inferencia.predict(datos[config["MODEL_FEATURES"]])
```

## Organización del proyecto

```
├── LICENSE                <- Licencia MIT
├── Makefile               <- Comandos útiles, como `make create_environment` o `make lint`
├── README.md              <- Este archivo
├── data                   <- No se versiona
│   ├── external           <- Datos de fuentes de terceros
│   ├── interim            <- Datos intermedios
│   ├── processed          <- Datasets transformados y predicciones de test
│   └── raw                <- Datos originales de Kaggle, inmutables
│
├── docs                   <- Proyecto de mkdocs
│
├── environment.yml        <- Entorno de conda env_house_pricing_a
│
├── models                 <- Pipelines y modelos entrenados (.joblib)
│
├── notebooks              <- Notebooks numerados en orden de ejecución, con la convención
│                             número, iniciales y descripción, p. ej. `1.0-bz-analisis-exploratorio`
│
├── pipeline_config.yaml   <- Features, parámetros, rutas y resultados del flujo
│
├── pyproject.toml         <- Metadatos del paquete clase_2 y configuración de ruff
│
├── references             <- Diccionarios de datos y material explicativo
│
├── reports                <- Resultados generados por los notebooks (no se versionan)
│   └── figures            <- Gráficas para reportes
│
├── requirements.txt       <- Dependencias del proyecto con versiones fijas
│
└── clase_2                <- Código fuente del proyecto (plantilla de cookiecutter)
    ├── __init__.py
    ├── config.py          <- Rutas y configuración compartida
    ├── dataset.py         <- Scripts para descargar o generar datos
    ├── features.py        <- Código para crear features
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py     <- Inferencia con modelos entrenados
    │   └── train.py       <- Entrenamiento de modelos
    └── plots.py           <- Visualizaciones
```

--------
