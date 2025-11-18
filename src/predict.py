# Código de Scoring - Modelo de Fuga de Colaboradores
############################################################################

import pandas as pd
import pickle
import os

path = "../data/scores/"
os.makedirs(path, exist_ok=True)

# -------------------------------------------------------------
# 1. Función de scoring
# -------------------------------------------------------------
def score_model(filename, scores):

    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('EMP_ID')
    print(filename, ' cargado correctamente')

    # ---------------------------------------------------------
    # 2. Cargar el modelo entrenado
    # ---------------------------------------------------------
    package = '../models/attrition_best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print("Modelo importado correctamente")

    # ---------------------------------------------------------
    # 3. Generar predicciones (probabilidad de fuga)
    # ---------------------------------------------------------
    res = model.predict_proba(df)[:, 1]   # Probabilidad de fuga
    pred = pd.DataFrame(res, columns=['PROB_FUGA'])

    # ---------------------------------------------------------
    # 4. Exportar archivo final
    # ---------------------------------------------------------
    pred.to_csv(os.path.join('../data/scores/', scores), index=False)
    print(scores, 'exportado correctamente en la carpeta scores')


# -------------------------------------------------------------
# 5. Ejecución del scoring
# -------------------------------------------------------------
def main():
    score_model('attrition_score.csv', 'attrition_final_score.csv')
    print("Finalizó el Scoring del Modelo de Fuga de Colaboradores")


if __name__ == "__main__":
    main()