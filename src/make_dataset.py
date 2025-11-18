# ===========================================================
# Script de Preparación de Datos - Fuga de Colaboradores
# ===========================================================

import pandas as pd
import numpy as np
import os

path = "../data/processed/"
os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------
# 1. Lectura de archivos CSV
# -----------------------------------------------------------
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('EMP_ID')
    print(filename, ' cargado correctamente')
    return df


# -----------------------------------------------------------
# 2. Transformación de Datos
# -----------------------------------------------------------

def data_preparation(df):

    # ==== 2.1 LIMPIEZA BÁSICA ====
    # Convertir género a binario: M=1 / F=0
    if "GENERO" in df.columns:
        df["GENERO"] = df["GENERO"].replace({"M": 1, "F": 0})

    # Reemplazar valores nulos en salarios, ausencias, desempeño
    df["SALARIO"] = df["SALARIO"].fillna(df["SALARIO"].median())
    df["AUSENCIAS"] = df["AUSENCIAS"].fillna(0)
    df["SATISFACCION"] = df["SATISFACCION"].fillna(df["SATISFACCION"].mean())
    df["DESEMPENO"] = df["DESEMPENO"].fillna(df["DESEMPENO"].median())
    df["ANTIGUEDAD_MESES"] = df["ANTIGUEDAD_MESES"].fillna(0)


    # ==== 2.2 TRANSFORMACIONES NUMÉRICAS (LOGS) ====
    df["LOG_SALARIO"] = np.log1p(df["SALARIO"])
    df["LOG_AUSENCIAS"] = np.log1p(df["AUSENCIAS"])
    df["LOG_ANTIGUEDAD"] = np.log1p(df["ANTIGUEDAD_MESES"])


    # ==== 2.3 VARIABLES DERIVADAS ====

    # Ratio de salario respecto al promedio del área
    if "AREA" in df.columns:
        df["SALARIO_AREA_MEAN"] = df.groupby("AREA")["SALARIO"].transform("mean")
        df["SALARIO_RATIO"] = df["SALARIO"] / (df["SALARIO_AREA_MEAN"] + 1)

    # Antigüedad normalizada
    df["ANTIGUEDAD_ANIOS"] = df["ANTIGUEDAD_MESES"] / 12

    # Variabilidad del desempeño (si existen evaluaciones históricas)
    eval_cols = [c for c in df.columns if "EVAL_" in c]  
    if len(eval_cols) > 1:
        df["STD_DESEMPENO"] = df[eval_cols].std(axis=1)
        df["AVG_DESEMPENO"] = df[eval_cols].mean(axis=1)
    else:
        df["STD_DESEMPENO"] = 0
        df["AVG_DESEMPENO"] = df["DESEMPENO"]

    # Interacción satisfacción + desempeño
    df["SATISFACCION_X_DESEMPENO"] = df["SATISFACCION"] * df["DESEMPENO"]

    # Indicador de burnout simple
    df["BURNOUT_SCORE"] = (df["AUSENCIAS"] > 5).astype(int) + (df["SATISFACCION"] < 2).astype(int)

    # Rotación histórica en el área (si viene en el dataset)
    if "ROTACION_AREA" in df.columns:
        df["RIESGO_AREA"] = df["ROTACION_AREA"] / 100


    # ==== 2.4 Dummies de variables categóricas ====
    cat_cols = ["AREA", "ESTADO_CIVIL", "CARGO"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("SIN_INFO")
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

    print("Transformación de datos completa")
    return df


# -----------------------------------------------------------
# 3. Exportación de datos procesados
# -----------------------------------------------------------

def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# -----------------------------------------------------------
# 4. MAIN – Generación de matrices finales (train, val, score)
# -----------------------------------------------------------

def main():

    # =================== TRAIN ======================
    df1 = read_file_csv('fuga_train.csv')
    tdf1 = data_preparation(df1)

    features_train = [
        "GENERO", "EDAD", "ANTIGUEDAD_MESES", "ANTIGUEDAD_ANIOS",
        "SALARIO", "LOG_SALARIO", "SALARIO_RATIO",
        "AUSENCIAS", "LOG_AUSENCIAS",
        "SATISFACCION", "DESEMPENO",
        "AVG_DESEMPENO", "STD_DESEMPENO",
        "SATISFACCION_X_DESEMPENO",
        "BURNOUT_SCORE",
        "LOG_ANTIGUEDAD",
        "FUGA"   # variable objetivo
    ] + [c for c in tdf1.columns if "AREA_" in c] \
      + [c for c in tdf1.columns if "ESTADO_CIVIL_" in c] \
      + [c for c in tdf1.columns if "CARGO_" in c]

    data_exporting(tdf1, features_train, 'attrition_train.csv')


    # =================== VALIDACIÓN ======================
    df2 = read_file_csv('fuga_val.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, features_train, 'attrition_val.csv')


    # =================== SCORING ======================
    df3 = read_file_csv('fuga_score.csv')
    tdf3 = data_preparation(df3)

    features_score = [f for f in features_train if f != "FUGA"]
    data_exporting(tdf3, features_score, 'attrition_score.csv')


# -----------------------------------------------------------
if __name__ == "__main__":
    main()