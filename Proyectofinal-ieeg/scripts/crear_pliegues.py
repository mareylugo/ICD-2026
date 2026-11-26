"""Quita fragmentos duplicados y asigna pliegues de validación cruzada agrupados por paciente.

Uso (desde la raíz del repositorio, después de extraer_atributos.py):
    python scripts/crear_pliegues.py

Salida:
    datos/pliegues.csv   hospital, segment_id, patient_id, category_name, pliegue

Los pliegues se guardan en un archivo para que el notebook principal y el notebook de la
CNN (Colab) usen exactamente la misma partición: así la comparación entre modelos es justa.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEMILLA = 42
N_PLIEGUES = 5


def pliegues_hospital(hospital):
    """Devuelve los fragmentos únicos de un hospital con su pliegue asignado."""
    segmentos = pd.read_csv(f"datos/segments_{hospital}.csv")
    huellas = pd.read_csv(f"datos/atributos_{hospital}.csv.gz", usecols=["segment_id", "md5"])
    df = segmentos.merge(huellas, on="segment_id").sort_values("segment_id")

    # Señales idénticas (mismo md5): se conserva la primera copia. Si una copia cayera en
    # entrenamiento y su gemela en prueba, el modelo acertaría por memoria.
    n_antes = len(df)
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
    print(f"{hospital}: {n_antes - len(df)} duplicados eliminados, {len(df)} fragmentos únicos")

    # Estratificado por clase y agrupado por paciente: ningún paciente queda en dos pliegues
    cv = StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    df["pliegue"] = -1
    for k, (_, idx_prueba) in enumerate(cv.split(df, df["category_name"], df["patient_id"])):
        df.loc[idx_prueba, "pliegue"] = k
    df["hospital"] = hospital
    return df[["hospital", "segment_id", "patient_id", "category_name", "pliegue"]]


if __name__ == "__main__":
    pliegues = pd.concat([pliegues_hospital(h) for h in ["fnusa", "mayo"]], ignore_index=True)
    # Verificación: cada paciente aparece en un solo pliegue
    assert (pliegues.groupby(["hospital", "patient_id"])["pliegue"].nunique() == 1).all()
    Path("datos").mkdir(exist_ok=True)
    pliegues.to_csv("datos/pliegues.csv", index=False)
    print(pliegues.groupby(["hospital", "pliegue"])["patient_id"].nunique().unstack())
