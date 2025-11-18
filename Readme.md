# 🚀 MLOps Pipeline — Modelo de Fuga de Colaboradores (Attrition)

Este proyecto implementa un **pipeline completo de MLOps** para predecir la fuga de colaboradores usando un modelo XGBoost.  
Incluye:

- Preparación de datos  
- Entrenamiento  
- Evaluación  
- Scoring  
- Exportación de modelo  

data/raw/
├── fuga_train.csv
├── fuga_val.csv
└── fuga_score.csv


---

# 📁 Estructura del Repositorio

````cs
│
├── data/
│ ├── raw/ # Datos originales (train / val / score)
│ ├── processed/ # Datos transformados para modelado
│ └── scores/ # Resultados de scoring
│
├── models/
│ └── attrition_best_model.pkl # Modelo entrenado
│
├── src/
│ ├── prepare_attrition_data.py # Feature engineering
│ ├── train_attrition_model.py # Entrenamiento con XGBoost
│ ├── eval_attrition_model.py # Evaluación del modelo
│ └── score_attrition_model.py # Scoring final
│
├── requirements.txt
├── .gitignore
└── README.md
````

---

# ⚙️ Instalación del Entorno

### Usando conda:

```bash
conda env create -f environment.yml
conda activate attrition-mlops

### Usando pip:
```bash
pip install -r requirements.txt

🧪 Ejecución del Pipeline (Ejemplos en Terminal)

Ejecutar desde la carpeta raíz del proyecto.

### 1. Preparación de Datos
```bash
python src/make_dataset.py

### 2. Entrenamiento del Modelo
```bash
python src/train.py

### 3. Evaluación del Modelo
```bash
python src/evaluate.py

### 4. Scoring Final
```bash
python src/predict.py




