# Código de Entrenamiento - Modelo de Fuga de Colaboradores
###########################################################################

import pandas as pd
import xgboost as xgb
import pickle
import os

path = "../models/"
os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# 1. Cargar la tabla transformada y entrenar el modelo
# ---------------------------------------------------------
def read_file_csv(filename):

    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('EMP_ID')
    print(filename, ' cargado correctamente')

    # Separar variables predictoras y target
    X_train = df.drop(['FUGA'], axis=1)
    y_train = df[['FUGA']]

    # -----------------------------------------------------
    # 2. Entrenamiento del modelo XGBoost
    # -----------------------------------------------------
    xgb_mod = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=120,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        seed=42,
        silent=True
    )

    xgb_mod.fit(X_train, y_train)
    print("Modelo entrenado correctamente")



    # -----------------------------------------------------
    # 3. Guardar el modelo para producción
    # -----------------------------------------------------
    package = '../models/attrition_best_model.pkl'
    pickle.dump(xgb_mod, open(package, 'wb'))

    print("Modelo exportado correctamente en la carpeta models")


# ---------------------------------------------------------
# 4. Ejecución del entrenamiento
# ---------------------------------------------------------
def main():
    read_file_csv('attrition_train.csv')
    print("Finalizó el entrenamiento del Modelo de Fuga de Colaboradores")


if __name__ == "__main__":
    main()
