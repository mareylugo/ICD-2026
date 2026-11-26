"""Intentos de mejora de los modelos de atributos (FNUSA, mismos pliegues por paciente).

Uso (desde la raíz del repositorio, después de crear_pliegues.py):
    python scripts/experimentos_mejora.py            # los tres experimentos (~2 h con 2 núcleos)
    python scripts/experimentos_mejora.py --exp 1 3  # solo algunos

Experimentos (métrica principal: F1 de la clase patológica):
    1. Normalización de los atributos por paciente (z-score con los fragmentos del mismo
       paciente; no usa etiquetas), sola o junto con los atributos originales.
    2. Umbral de decisión para la clase patológica, elegido con 3 pliegues internos por
       paciente dentro del entrenamiento de cada pliegue externo.
    3. Hiperparámetros con validación anidada (3 pliegues internos por paciente) para
       gradient boosting y SVM, y SVM entrenado con 30 mil fragmentos en lugar de 10 mil.

Salida (la carga el notebook principal, sección 5.4):
    resultados/mejoras_pliegues.csv          F1 por pliegue de cada variante
    resultados/mejoras_predicciones.csv.gz   predicciones fuera de pliegue de cada variante
    resultados/mejoras_detalle.csv           umbrales e hiperparámetros elegidos por pliegue

Este script es costoso y se ejecutó aparte; la configuración de los modelos y la
transformación de atributos son idénticas a las del notebook principal (secciones 3 y 4).
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEMILLA = 42
RESULTADOS = Path("resultados")
MAPA_CLASES = {"physiology": "fisiologica", "pathology": "patologica", "noise": "artefacto",
               "powerline_50hz": "ruido_linea", "powerline_60hz": "ruido_linea"}

# Modelos con la misma configuración que en el notebook principal
MODELOS = {
    "Regresión logística": make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, class_weight="balanced", random_state=SEMILLA)),
    "Random forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEMILLA),
    "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                                        early_stopping=False, random_state=SEMILLA),
}
N_ENTRENAMIENTO = {"SVM (RBF)": 10_000}      # submuestra de entrenamiento, como en el notebook


# ---------------------------------------------------------------- datos

def cargar_fnusa():
    """X (atributos con log10 en los de cola larga), y, paciente y pliegue de FNUSA (sin duplicados)."""
    atributos = pd.read_csv("datos/atributos_fnusa.csv.gz")
    pliegues = pd.read_csv("datos/pliegues.csv").query("hospital == 'fnusa'")
    df = pliegues.merge(atributos, on="segment_id", validate="one_to_one")
    columnas = [c for c in atributos.columns if c not in ["segment_id", "md5", "media", "desv_est"]]
    con_log = [c for c in columnas if c.startswith(("pot_rel_", "cociente_", "env_max_")) or c == "planitud_espectral"]
    X = df[columnas].astype(float).copy()
    X[con_log] = np.log10(X[con_log] + 1e-8)
    return (X.to_numpy(), df["category_name"].map(MAPA_CLASES).to_numpy(), df["patient_id"].to_numpy(),
            df["pliegue"].to_numpy(), df["segment_id"].to_numpy())


# ---------------------------------------------------------------- utilidades (iguales al notebook)

def f1_patologica(y, pred):
    """F1 de la clase patológica."""
    return f1_score(y, pred, labels=["patologica"], average=None, zero_division=0)[0]


def submuestra_estratificada(idx, y, n):
    """Toma n índices de idx conservando la proporción de clases (semilla fija)."""
    if not n or len(idx) <= n:
        return idx
    return (pd.Series(idx).groupby(y[idx], group_keys=False)
            .apply(lambda s: s.sample(frac=n / len(idx), random_state=SEMILLA)).to_numpy())


def norm_paciente(X, g):
    """z-score de cada atributo con la media y la desviación de los fragmentos del mismo paciente."""
    Z = np.empty_like(X)
    for p in np.unique(g):
        m = g == p
        sd = X[m].std(axis=0)
        Z[m] = (X[m] - X[m].mean(axis=0)) / np.where(sd > 1e-9, sd, 1)
    return Z


def margen_patologica(m, X):
    """Puntaje de la clase patológica menos el mayor de las demás clases."""
    s = m.predict_proba(X) if hasattr(m, "predict_proba") else m.decision_function(X)
    ip = list(m.classes_).index("patologica")
    return s[:, ip] - np.delete(s, ip, axis=1).max(axis=1), s


def predecir_con_umbral(m, X, t):
    """Predice patológica si su margen supera t; si no, la más probable de las demás clases."""
    mg, s = margen_patologica(m, X)
    clases = list(m.classes_)
    otras = [c for c in clases if c != "patologica"]
    pred = np.array(otras)[np.delete(s, clases.index("patologica"), axis=1).argmax(axis=1)]
    pred[mg >= t] = "patologica"
    return pred


# ---------------------------------------------------------------- registro de resultados

class Registro:
    """Acumula F1 por pliegue y predicciones fuera de pliegue de cada variante y los guarda."""

    def __init__(self, n, segment_id):
        self.pliegues, self.detalle, self.pred = [], [], {"segment_id": segment_id}
        self.n = n

    def agregar(self, experimento, modelo, variante, y, k, pred):
        for j in np.unique(k):
            self.pliegues.append({"experimento": experimento, "modelo": modelo, "variante": variante, "pliegue": j,
                                  "f1_patologica": f1_patologica(y[k == j], pred[k == j]),
                                  "f1_macro": f1_score(y[k == j], pred[k == j], average="macro")})
        self.pred[f"{experimento} | {modelo} | {variante}"] = pred
        f = [r["f1_patologica"] for r in self.pliegues if r["modelo"] == modelo and r["variante"] == variante
             and r["experimento"] == experimento]
        print(f"[{experimento}] {modelo} | {variante}: F1 patológica = {np.mean(f):.3f} ± {np.std(f):.3f} "
              f"(global {f1_patologica(y, pred):.3f})", flush=True)
        self.guardar()

    def guardar(self):
        RESULTADOS.mkdir(exist_ok=True)
        pd.DataFrame(self.pliegues).to_csv(RESULTADOS / "mejoras_pliegues.csv", index=False)
        pd.DataFrame(self.detalle).to_csv(RESULTADOS / "mejoras_detalle.csv", index=False)
        pd.DataFrame(self.pred).to_csv(RESULTADOS / "mejoras_predicciones.csv.gz", index=False)


def validar(modelo, X, y, k, n_ent=None):
    """Predicciones fuera de pliegue con los pliegues fijos (como evaluar_cv del notebook)."""
    pred = np.empty(len(y), dtype=object)
    for j in np.unique(k):
        idx = submuestra_estratificada(np.flatnonzero(k != j), y, n_ent)
        pred[k == j] = clone(modelo).fit(X[idx], y[idx]).predict(X[k == j])
    return pred


# ---------------------------------------------------------------- experimentos

def experimento_1(X, y, g, k, reg):
    """Normalización por paciente."""
    Z = norm_paciente(X, g)
    variantes = {"original": X, "normalizado por paciente": Z, "original + normalizado": np.hstack([X, Z])}
    for nombre in ["Regresión logística", "SVM (RBF)", "Gradient boosting", "Random forest"]:
        for var, Xv in variantes.items():
            reg.agregar("1. normalización por paciente", nombre, var, y, k,
                        validar(MODELOS[nombre], Xv, y, k, N_ENTRENAMIENTO.get(nombre)))


def elegir_umbral(modelo, X, y, g, n_ent):
    """Umbral que maximiza la F1 patológica en 3 pliegues internos por paciente."""
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=SEMILLA)
    margenes, reales = [], []
    for a, b in cv.split(X, y, g):
        ia = submuestra_estratificada(a, y, n_ent)
        ib = submuestra_estratificada(b, y, 30_000)
        m = clone(modelo).fit(X[ia], y[ia])
        margenes.append(margen_patologica(m, X[ib])[0])
        reales.append(y[ib])
    margenes, reales = np.concatenate(margenes), np.concatenate(reales)
    candidatos = np.append(np.quantile(margenes, np.linspace(0.02, 0.98, 97)), 0.0)
    f1s = [f1_score(reales == "patologica", margenes >= t) for t in candidatos]
    return candidatos[int(np.argmax(f1s))]


def experimento_2(X, y, g, k, reg):
    """Umbral de decisión para la clase patológica."""
    for nombre in ["Regresión logística", "SVM (RBF)", "Random forest"]:
        n_ent = N_ENTRENAMIENTO.get(nombre)
        pred_base, pred_umbral = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
        for j in np.unique(k):
            tr, te = np.flatnonzero(k != j), k == j
            t = elegir_umbral(MODELOS[nombre], X[tr], y[tr], g[tr], n_ent)
            idx = submuestra_estratificada(tr, y, n_ent)
            m = clone(MODELOS[nombre]).fit(X[idx], y[idx])
            pred_base[te] = m.predict(X[te])
            pred_umbral[te] = predecir_con_umbral(m, X[te], t)
            reg.detalle.append({"experimento": "2. umbral", "modelo": nombre, "pliegue": j, "elegido": f"{t:+.3f}"})
        reg.agregar("2. umbral", nombre, "clase más probable", y, k, pred_base)
        reg.agregar("2. umbral", nombre, "umbral elegido por validación interna", y, k, pred_umbral)


REJILLAS = {
    "Gradient boosting": (
        lambda p: HistGradientBoostingClassifier(
            max_iter=p["max_iter"], learning_rate=0.1, max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=1.0, class_weight="balanced",
            early_stopping=False, random_state=SEMILLA),
        {"max_depth": [3, 6], "min_samples_leaf": [100, 1000], "max_iter": [100, 300]}, 50_000, None),
    "SVM (RBF)": (
        lambda p: make_pipeline(StandardScaler(), SVC(C=p["C"], gamma=p["gamma"], class_weight="balanced",
                                                      random_state=SEMILLA)),
        {"C": [0.3, 1.0, 3.0], "gamma": [0.01, 0.033, 0.1]}, 10_000, 10_000),
}


def experimento_3(X, y, g, k, reg):
    """Hiperparámetros con validación anidada y SVM con más datos."""
    for nombre, (crear, rejilla, n_interno, n_externo) in REJILLAS.items():
        combinaciones = [dict(zip(rejilla, v)) for v in itertools.product(*rejilla.values())]
        pred = np.empty(len(y), dtype=object)
        for j in np.unique(k):
            tr, te = np.flatnonzero(k != j), k == j
            internos = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEMILLA).split(X[tr], y[tr], g[tr]))
            puntajes = []
            for p in combinaciones:
                f = []
                for a, b in internos:
                    ia = submuestra_estratificada(tr[a], y, n_interno)
                    ib = submuestra_estratificada(tr[b], y, 20_000)
                    f.append(f1_patologica(y[ib], crear(p).fit(X[ia], y[ia]).predict(X[ib])))
                puntajes.append(np.mean(f))
            elegido = combinaciones[int(np.argmax(puntajes))]
            reg.detalle.append({"experimento": "3. hiperparámetros", "modelo": nombre, "pliegue": j,
                                "elegido": str(elegido)})
            idx = submuestra_estratificada(tr, y, n_externo)
            pred[te] = crear(elegido).fit(X[idx], y[idx]).predict(X[te])
        reg.agregar("3. hiperparámetros", nombre, "valores por omisión",
                    y, k, validar(MODELOS[nombre], X, y, k, N_ENTRENAMIENTO.get(nombre)))
        reg.agregar("3. hiperparámetros", nombre, "validación anidada", y, k, pred)
    reg.agregar("3. hiperparámetros", "SVM (RBF)", "entrenado con 30 mil fragmentos",
                y, k, validar(MODELOS["SVM (RBF)"], X, y, k, 30_000))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    X, y, g, k, sid = cargar_fnusa()
    reg = Registro(len(y), sid)
    t0 = time.time()
    for e in args.exp:
        {1: experimento_1, 2: experimento_2, 3: experimento_3}[e](X, y, g, k, reg)
        print(f"Experimento {e} terminado ({(time.time() - t0) / 60:.0f} min)", flush=True)
