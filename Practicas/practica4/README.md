# Práctica 4 · Ensambles: deserción estudiantil y Bank Marketing

¿Cuánto ganan los ensambles (*bagging*, *boosting*, votación y *stacking*) sobre clasificadores que ya están bien ajustados? Se responde en dos problemas:

1. **Deserción estudiantil** (Práctica 3): los cinco clasificadores base de la Práctica 3 (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística) se recuperan con su mejor preprocesamiento e hiperparámetros, sin volver a optimizarlos, y se comparan con *bagging*, AdaBoost, votación dura y suave, *stacking*, Random Forest, Extra-Trees, XGBoost y LightGBM, sobre los mismos cinco pliegues estratificados.
2. **Bank Marketing** (Tarea 4): búsqueda del mejor modelo con todo lo visto en el curso (preprocesamiento, balanceo de clases, clasificadores, optimización y ensambles), evaluado con `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` y F1 macro.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `practica4_ensambles.ipynb` | Notebook completo, con todas las salidas: las dos partes y las conclusiones |
| `marey-lugo-lightgbm.joblib` | Mejor modelo de Bank Marketing (LightGBM optimizado con ajuste de umbral), entrenado con los 41 188 clientes; lo genera el notebook |
| `datos/desercion-entrenamiento.csv` | Conjunto de deserción (el mismo de la Práctica 3) |
| `datos/bank-additional-full.csv` | Conjunto Bank Marketing |
| `requirements.txt` | Librerías y versiones utilizadas |

## Datos

### Deserción estudiantil (`datos/desercion-entrenamiento.csv`)

3539 estudiantes de licenciatura del Instituto Politécnico de Portalegre (Portugal) que ingresaron entre 2008/09 y 2018/19, con su situación al final de la duración normal de la carrera: `Graduate` (49.9 %), `Dropout` (32.1 %) o `Enrolled` (17.9 %). Como en la Práctica 3, se usan solo los **30 atributos conocidos al cierre del primer semestre**: datos demográficos, socioeconómicos y académicos al inscribirse (9 nominales codificados como enteros, 8 binarios, 1 ordinal y 3 numéricos), el desempeño del primer semestre (6 numéricos) y tres indicadores macroeconómicos. Los seis atributos del segundo semestre no se usan. No hay valores faltantes.

Proviene del repositorio del curso ([husseinlopez/icd2026](https://github.com/husseinlopez/icd2026), carpeta `datos/`), que lo tomó de *Predict Students' Dropout and Academic Success* (Realinho *et al.*, 2022; [UCI 697](https://archive.ics.uci.edu/dataset/697), licencia CC BY 4.0).

### Bank Marketing (`datos/bank-additional-full.csv`)

41 188 llamadas de campañas de telemarketing de un banco portugués (mayo de 2008 a noviembre de 2010), separadas por `;`. La variable objetivo `y` indica si el cliente contrató un depósito a plazo: *yes* en el 11.3 % de los casos. Tiene 20 atributos: datos del cliente (edad, empleo, estado civil, educación, crédito, préstamos), del contacto (medio, mes, día de la semana, número de contactos, días desde el contacto anterior y resultado de la campaña anterior) y cinco indicadores socioeconómicos. Varios atributos categóricos tienen un nivel `unknown`. **Se quita `duration`**, porque la duración de la llamada solo se conoce después de llamar, así que quedan 19 atributos. El objetivo se codifica como 1 (*yes*) y 0 (*no*).

Fuente: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222), https://doi.org/10.24432/C5K306, licencia CC BY 4.0. El archivo es `bank-additional/bank-additional-full.csv` dentro del zip de UCI.

## Requisitos

- Python 3.10 o posterior (se ejecutó con Python 3.13)
- Las librerías de `requirements.txt`: numpy, pandas, scipy, scikit-learn (≥ 1.5, por `TunedThresholdClassifierCV`; se probó con 1.9.1), matplotlib, joblib, xgboost, lightgbm, jinja2 (para las tablas con formato) y Jupyter.

## Uso

```bash
git clone <URL de este repositorio>
cd <carpeta del repositorio>
pip install -r requirements.txt
jupyter notebook practica4_ensambles.ipynb
```

En Jupyter, ejecuta **Kernel → Restart & Run All**. Todas las rutas son relativas a la raíz del repositorio. Las semillas están fijas: `SEMILLA = 42` en la deserción (la misma de la Práctica 3, de la que dependen los pliegues) y `SEMILLA_BANCO = 2026` en Bank Marketing. La ejecución completa tarda unos 40 minutos en una computadora con 2 núcleos; las evaluaciones y búsquedas usan todos los núcleos disponibles. Al terminar, el notebook escribe el `.joblib` y verifica que, al cargarlo, reproduce el F1 macro reportado.

Para usar el modelo de Bank Marketing:

```python
import joblib, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

modelo = joblib.load("marey-lugo-lightgbm.joblib")
datos = pd.read_csv("datos/bank-additional-full.csv", sep=";")
X = datos.drop(columns=["duration", "y"])
y = (datos["y"] == "yes").astype(int)

modelo.predict(X)                       # 1 = se recomienda llamar (umbral ya ajustado)
cross_validate(modelo, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026),
               scoring="f1_macro")["test_score"].mean()     # ≈ 0.721
```

El modelo está formado solo por clases de scikit-learn y LightGBM (`TunedThresholdClassifierCV` → `Pipeline` con `ColumnTransformer` y `LGBMClassifier`), así que se carga sin definir nada adicional.

## Resultados principales

**Deserción** (F1 macro, 5 pliegues; mejor base: SVM con 0.682 ± 0.010):

| Modelo | F1 macro |
|---|---|
| XGBoost optimizado | 0.691 ± 0.016 |
| Votación dura de los 5 base | 0.688 ± 0.009 |
| Random Forest optimizado | 0.684 ± 0.013 |
| LightGBM optimizado | 0.684 ± 0.010 |
| *Stacking* de los 5 base | 0.683 ± 0.011 |
| SVM (Práctica 3) | 0.682 ± 0.010 |

**Bank Marketing** (F1 macro con los pliegues del enunciado):

| Modelo | F1 macro | Sensibilidad de *yes* |
|---|---|---|
| LightGBM optimizado + ajuste de umbral | 0.721 ± 0.006 | 0.547 |
| LightGBM + ajuste de umbral | 0.720 ± 0.007 | 0.530 |
| Random Forest + ajuste de umbral | 0.719 ± 0.007 | 0.536 |
| Regresión logística + ajuste de umbral | 0.709 ± 0.006 | 0.520 |

En los dos problemas, los ensambles ganan poco sobre modelos bien ajustados: +0.009 en la deserción y +0.012 sobre la regresión logística en Bank Marketing. Lo decisivo fue la regla de decisión: los pesos por clase en la deserción y el ajuste del umbral en Bank Marketing, donde subió el F1 macro del mismo LightGBM de 0.657 a 0.720. La discusión completa está en las secciones 13 y 14 del notebook.

## Declaración de uso de IA

Se utilizó el asistente de inteligencia artificial Claude (Anthropic) en la elaboración de este notebook. La autora revisó el contenido y es responsable de su exactitud.
