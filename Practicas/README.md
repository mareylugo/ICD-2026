# Práctica 3 · Predicción de la deserción estudiantil

Clasificador que predice, al cierre del primer semestre, la situación de un estudiante al final de la duración normal de su carrera: **graduado** (`graduate`), **desertor** (`dropout`) o **todavía inscrito** (`enrolled`). Se comparan cinco clasificadores optimizados (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística), todos con los mismos pliegues de validación cruzada estratificada, y se usa el F1 macro como métrica principal.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `practica3_desercion.ipynb` | Notebook con el análisis completo: descripción de los datos, preprocesamiento, búsqueda de hiperparámetros, comparación, análisis de errores, estimación con el segundo semestre, modelo final y conclusiones |
| `datos/desercion-entrenamiento.csv` | 3539 estudiantes, 36 atributos y `Target` |
| `datos/desercion-prueba.csv` | 885 estudiantes, solo con los 30 atributos conocidos al cierre del primer semestre |
| `predicciones.csv` | Predicciones del clasificador final para los 885 estudiantes de prueba (columnas `id`, `Target`); lo genera el notebook |
| `requirements.txt` | Librerías y versiones utilizadas |

## Datos

4424 estudiantes de licenciatura del Instituto Politécnico de Portalegre (Portugal) que ingresaron entre los ciclos 2008/09 y 2018/19. Cada estudiante se describe con:
- datos demográficos, socioeconómicos y académicos al inscribirse;
- su desempeño en el primer y el segundo semestres (unidades acreditadas, inscritas, evaluadas, aprobadas, promedio y sin evaluación);
- tres indicadores macroeconómicos (desempleo, inflación y PIB).

Los nueve atributos nominales vienen codificados como enteros. La variable objetivo es `Target` (Graduate 49.9 %, Dropout 32.1 %, Enrolled 17.9 % en el entrenamiento).

Los archivos provienen del repositorio del curso ([husseinlopez/icd2026](https://github.com/husseinlopez/icd2026), carpeta `datos/`). Son una partición estratificada del conjunto *Predict Students' Dropout and Academic Success* (Realinho *et al.*, 2022), distribuido en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697) con licencia CC BY 4.0. La ficha de UCI describe cada atributo y el significado de sus códigos.

## Requisitos

- Python 3.10 o posterior (se ejecutó con Python 3.13)
- Las librerías de `requirements.txt`: numpy, pandas, scipy, scikit-learn, matplotlib y Jupyter. Se probó con scikit-learn 1.9.1; con versiones anteriores a 1.8, el notebook ajusta automáticamente la forma de pedir la regularización de la regresión logística.

## Uso

```bash
git clone <URL de este repositorio>
cd <carpeta del repositorio>
pip install -r requirements.txt
jupyter notebook practica3_desercion.ipynb
```

En Jupyter, ejecuta **Kernel → Restart & Run All**. Todas las rutas son relativas a la raíz del repositorio y la semilla está fijada (`SEMILLA = 42`). La ejecución completa tarda unos 8 minutos en una computadora con 2 núcleos; las búsquedas usan todos los núcleos disponibles. Al terminar, el notebook escribe y verifica `predicciones.csv`.

## Resultados principales

F1 macro de validación cruzada (5 pliegues, mismos pliegues para todos):

| Clasificador | F1 macro |
|---|---|
| SVM (kernel polinomial, grado 3) | 0.682 ± 0.010 |
| Regresión logística (*elastic net*) | 0.680 ± 0.010 |
| Árbol de decisión | 0.648 ± 0.014 |
| Naive Bayes (categórico) | 0.627 ± 0.015 |
| kNN | 0.574 ± 0.018 |

El clasificador final es el SVM. Si se esperara al cierre del segundo semestre, su F1 macro subiría a 0.716, pero la intervención se retrasaría un semestre; la discusión está en la sección 9 del notebook.

## Declaración de uso de IA

Se utilizó el asistente de inteligencia artificial Claude (Anthropic) en la elaboración de este notebook. La autora revisó el contenido y es responsable de su exactitud.
