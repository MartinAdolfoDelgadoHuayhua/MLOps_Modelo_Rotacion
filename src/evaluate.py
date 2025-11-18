# Código de Evaluación - Modelo de Fuga de Colaboradores

import pandas as pd
import pickle
from sklearn.metrics import *
import os


# ---------------------------------------------------------
# 1. Función de Evaluación
# ---------------------------------------------------------
def eval_model(filename):

    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('EMP_ID')
    print(filename, ' cargado correctamente')

    # -----------------------------------------------------
    # 2. Cargar modelo entrenado
    # -----------------------------------------------------
    package = '../models/attrition_best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print("Modelo importado correctamente")

    # -----------------------------------------------------
    # 3. Preparar datos de validación
    # -----------------------------------------------------
    X_test = df.drop(['FUGA'], axis=1)
    y_test = df[['FUGA']]

    # Predicción clases
    y_pred = model.predict(X_test)

    # Predicción probabilidades (para AUC)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # 4. Métricas de diagnóstico
    # -----------------------------------------------------
    cm_test = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(cm_test)

    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_prob)

    print("\nAccuracy: ", accuracy_test)
    print("Precision: ", precision_test)
    print("Recall: ", recall_test)
    print("F1 Score: ", f1_test)
    print("AUC ROC: ", auc_test)


# ---------------------------------------------------------
# 5. Ejecución completa
# ---------------------------------------------------------
def main():
    eval_model('attrition_val.csv')
    print("Finalizó la validación del Modelo de Fuga de Colaboradores")


if __name__ == "__main__":
    main()