# Clasificación automática de actividad patológica en EEG intracraneal

Proyecto final de **Introducción a la Ciencia de Datos** (CICESE, 2026) · Luz Marey Lugo Acedo

## Descripción

En la epilepsia que no responde a fármacos, la cirugía depende de localizar el tejido que genera las crisis con EEG intracraneal (iEEG). Hoy un experto revisa a mano días de registro, y los detectores automáticos confunden los artefactos musculares con la actividad patológica, porque ambos tienen energía en alta frecuencia.

El proyecto clasifica fragmentos de 3 s de iEEG en cuatro clases: **fisiológica**, **patológica** (espigas y oscilaciones de alta frecuencia, HFOs), **artefacto** y **ruido de línea**. Para ello calcula 30 atributos espectrales, temporales y de eventos, y compara líneas base, siete familias de modelos del curso, una red neuronal (MLP) y una CNN 1D sobre la señal cruda.

La sección 8 del notebook repite la comparación en el subconjunto de 40,539 fragmentos de FNUSA que usa la CNN, con los mismos pliegues, y agrega:

- 196 atributos calculados con librerías especializadas: **specparam** (componente 1/f y picos del espectro), **antropy** (entropías y dimensiones fractales), **MNE** (potencia *multitaper*), **EPyCom** (detectores publicados de HFO y espigas) y **TSFEL** (atributos estadísticos, temporales y espectrales).
- Modelos de *boosting* (**LightGBM** y **XGBoost**) y ajuste de hiperparámetros con **optuna** en validación anidada por paciente.
- Importancia de atributos con **SHAP** para P2.
- Redes publicadas para EEG con **braindecode**, frente a la CNN propia. El código admite ShallowFBCSPNet, EEGNet y Deep4Net; por tiempo y cómputo, en la versión entregada solo se entrenó EEGNet.

**Preguntas de investigación**

1. **P1.** ¿Bastan atributos espectrales y temporales para separar la actividad patológica de la fisiológica y de los artefactos? (F1 de la clase patológica con partición por paciente frente a las líneas base)
2. **P2.** ¿Qué atributos pesan más para distinguir lo patológico de los artefactos? (ranking de importancia estable entre pliegues)
3. **P3.** ¿Un modelo entrenado en un hospital conserva su desempeño en el otro? (F1 de la clase patológica en el otro hospital frente a la validación dentro del mismo)

## Datos

- **Origen:** Nejedly, P. et al. (2020). *Multicenter intracranial EEG dataset for classification of graphoelements and artifactual signals*. *Scientific Data* 7:179. [doi:10.1038/s41597-020-0532-5](https://doi.org/10.1038/s41597-020-0532-5)
- **Descarga:** colección de figshare [doi:10.6084/m9.figshare.c.4681208](https://doi.org/10.6084/m9.figshare.c.4681208). Se usan `Dataset_Fnusa.zip` (St. Anne's University Hospital, Brno; ~6.1 GB) y `Dataset_Mayo.zip` (Clínica Mayo; ~2.4 GB).
- **Licencia:** CC0.
- **Contenido:** 348,300 fragmentos de 3 s (15,000 muestras a 5 kHz) de 38 pacientes, cada uno en un `.mat`, más un `segments.csv` por hospital con metadatos y etiqueta.

Los zips originales **no están en el repositorio** (8.5 GB). Para reproducir todo desde cero, hay que descargarlos y colocarlos en `datos/crudos/`.

Los archivos derivados sí están en `datos/` (`atributos_*.csv.gz`, `segments_*.csv`, `pliegues.csv`, `cnn_subconjunto_ids.csv` y `atributos_extendidos_subconjunto.csv.gz`, de 57 MB; GitHub admite archivos de hasta 100 MB), así que el notebook se puede ejecutar sin los zips. El subconjunto de señales para la CNN (`cnn_subconjunto_fnusa.npz`, ~300 MB) no se sube: se genera con el paso 3.

## Requisitos

- Python 3.10 o superior.
- Notebook principal y scripts: `pip install -r requirements.txt`. Versiones con las que se probó: numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2.
- Los scripts de `scripts/` solo necesitan NumPy (y pandas y scikit-learn para `crear_pliegues.py`).
- La sección 7 (CNN) necesita TensorFlow y, en la práctica, una GPU: por eso el notebook se ejecuta en **Google Colab con GPU**, donde ya vienen todas las librerías. Fuera de Colab, si no hay TensorFlow, la sección 7 carga las predicciones guardadas en `resultados/cnn_predicciones.csv`.
- La sección 8 y `scripts/extraer_atributos_extendidos.py` usan las librerías de la segunda parte de `requirements.txt` (en Colab, la sección 8 instala las que faltan). specparam y EPyCom solo tienen versiones preliminares, por eso están fijadas. La subsección 8.6 (braindecode) necesita PyTorch: si `resultados/braindecode_predicciones.csv` ya tiene las redes elegidas en `REDES_BD`, las carga; si no, las entrena con GPU o, con `BD_EN_CPU = True`, en el procesador. En la versión entregada solo se entrenó EEGNet, en CPU (~1 h), porque se agotaron las unidades de GPU de Colab; ShallowFBCSPNet (~2 h en CPU) y Deep4Net (~6 h) quedaron fuera por tiempo.

## Orden de ejecución

Todos los comandos se ejecutan desde la raíz del repositorio.

| Paso | Comando | Tiempo aprox. | Genera |
|---|---|---|---|
| 1 | `python scripts/extraer_atributos.py --hospital fnusa --zip datos/crudos/Dataset_Fnusa.zip` y lo mismo con `--hospital mayo --zip datos/crudos/Dataset_Mayo.zip` | 18 + 15 min | `datos/atributos_*.csv.gz`, `datos/segments_*.csv` |
| 2 | `python scripts/crear_pliegues.py` | < 1 min | `datos/pliegues.csv` |
| 3 | `python scripts/preparar_subconjunto_cnn.py --zip datos/crudos/Dataset_Fnusa.zip` | 2 min | `datos/cnn_subconjunto_fnusa.npz`, `datos/cnn_subconjunto_ids.csv` |
| 4 | `python scripts/experimentos_mejora.py` (intentos de mejora; el notebook carga sus resultados en la sección 5.4) | ~2 h | `resultados/mejoras_*.csv` |
| 5 | `python scripts/extraer_atributos_extendidos.py` (atributos de la sección 8; necesita el `.npz` del paso 3; `--limite 200` hace una prueba rápida; si se interrumpe, al volver a ejecutarlo retoma las partes que faltan) | ~100 min con 14 procesos (i7 de 13.ª generación) | `datos/atributos_extendidos_subconjunto.csv.gz` |
| 6 | Subir la carpeta del proyecto a Google Drive como `MyDrive/proyecto_ieeg` (al menos `datos/`, con el `.npz` y los atributos extendidos, y `resultados/`), abrir `proyecto_ieeg.ipynb` en Colab, elegir *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4* y ejecutar todo (*Ejecutar todas*). Al final, *Archivo → Descargar → .ipynb* | Secciones 1–7: ~30 min (15 de modelos clásicos + 15 de CNN). Sección 8: 8.1–8.5 tardan ~20 min con 16 hilos y bastante más con los 2 núcleos de Colab (se puede bajar `N_INTENTOS_OPTUNA`); 8.6 (braindecode), ~1 h por EEGNet en CPU (medido) o mucho menos con GPU | notebook ejecutado, figuras, `resultados/*.csv`, `resultados/figuras_extensiones/` |

Los pasos 1 a 5 ya están hechos si se usan los archivos de `datos/` y `resultados/` del repositorio. Como el notebook completo puede pasar de 3 horas en Colab, conviene guardar el notebook en Drive de vez en cuando. Las predicciones de la CNN y de braindecode se guardan en `resultados/`; la sección 7 las carga en lugar de entrenar si no hay TensorFlow, y la 8.6 si no hay GPU.

## Estructura

```
proyecto_ieeg/
├── README.md
├── requirements.txt
├── proyecto_ieeg.ipynb            # notebook completo: datos, EDA, preprocesamiento, modelos, CNN, evaluación,
│                                  #   extensiones (librerías, LightGBM/XGBoost, optuna, SHAP, braindecode) y conclusiones
├── scripts/
│   ├── lectura_mat.py             # lectura de los .mat directamente desde los zips (solo NumPy)
│   ├── extraer_atributos.py       # 30 atributos por fragmento
│   ├── extraer_atributos_extendidos.py  # 196 atributos con specparam, antropy, MNE, EPyCom y TSFEL (subconjunto)
│   ├── crear_pliegues.py          # quita duplicados y asigna 5 pliegues agrupados por paciente
│   ├── preparar_subconjunto_cnn.py  # señales submuestreadas a 1250 Hz para la CNN
│   └── experimentos_mejora.py     # normalización por paciente, umbral e hiperparámetros (validación anidada)
├── datos/
│   ├── crudos/                    # (no incluido) Dataset_Fnusa.zip y Dataset_Mayo.zip
│   ├── atributos_fnusa.csv.gz, atributos_mayo.csv.gz
│   ├── segments_fnusa.csv, segments_mayo.csv
│   ├── pliegues.csv
│   ├── cnn_subconjunto_ids.csv
│   ├── atributos_extendidos_subconjunto.csv.gz  # se genera en el paso 5
│   └── cnn_subconjunto_fnusa.npz  # (no incluido, ~300 MB) se genera en el paso 3
└── resultados/                    # tablas, figuras y predicciones generadas por el notebook
```

## Decisiones principales

- **Partición por paciente** (`StratifiedGroupKFold`, 5 pliegues, semilla 42), compartida por todos los modelos, incluida la CNN.
- **Deep learning en el mismo notebook:** un MLP sobre los atributos y una CNN 1D sobre la señal cruda, comparada con el mejor modelo de atributos en el mismo subconjunto de fragmentos.
- **Duplicados** (señales idénticas) eliminados antes de partir.
- **Atributos relativos y de forma**, porque cada señal viene normalizada a desviación estándar 1.
- **Métrica principal: F1 de la clase patológica**, porque el objetivo clínico es detectar la actividad epileptiforme (combina falsos negativos y falsas alarmas). Se reportan su variabilidad entre pliegues y un intervalo de confianza por *bootstrap* de pacientes. Métrica secundaria: F1 macro.

## Declaración de uso de IA generativa

Se utilizó el asistente de inteligencia artificial Claude (Anthropic) en la elaboración de este proyecto. La autora revisó el contenido y es responsable de su exactitud.
