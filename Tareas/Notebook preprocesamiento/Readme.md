# Preprocesamiento del conjunto de datos Breast Cancer Wisconsin (Original)

Notebook de la materia **Introducción a la Ciencia de Datos 2026**, Posgrado en Ciencias de la Computación, CICESE.

**Autora:** Luz Marey Lugo Acedo

## Descripción

Este notebook explora y preprocesa el conjunto de datos *Breast Cancer Wisconsin (Original)*, que describe 699 muestras de citología de mama mediante nueve atributos numéricos (escala de 1 a 10) y un diagnóstico benigno o maligno.

El objetivo es mejorar la calidad de los datos antes de un posible modelado, aplicando dos técnicas:

1. **Limpieza de datos:** eliminación de registros duplicados e imputación de valores faltantes.
2. **Aumento de datos:** balanceo de clases con SMOTE.

## Contenido del repositorio

```
Tareas/
└── Notebook preprocesamiento/
    ├── Notebook_preprocesamiento.ipynb   # Notebook principal
    ├── breast-cancer-wisconsin.data      # Datos (sin encabezado)
    ├── breast-cancer-wisconsin.names     # Documentación del dataset
    └── README.md
```

## Conjunto de datos

| Columna | Descripción | Valores |
|---|---|---|
| `id` | Código de muestra (solo identificador) | entero |
| `clump_thickness` | Grosor del grumo | 1 a 10 |
| `cell_size` | Uniformidad del tamaño celular | 1 a 10 |
| `cell_shape` | Uniformidad de la forma celular | 1 a 10 |
| `marginal_adhesion` | Adhesión marginal | 1 a 10 |
| `epithelial_size` | Tamaño de célula epitelial | 1 a 10 |
| `bare_nuclei` | Núcleos desnudos | 1 a 10 (16 valores faltantes, marcados con `?`) |
| `bland_chromatin` | Cromatina blanda | 1 a 10 |
| `normal_nucleoli` | Nucléolos normales | 1 a 10 |
| `mitoses` | Mitosis | 1 a 10 |
| `class` | Diagnóstico | 2 = benigno, 4 = maligno (recodificado a 0 y 1 en el notebook) |

## Estructura del notebook

1. **Introducción:** contexto del cáncer de mama y descripción del dataset.
2. **Análisis exploratorio:** carga de datos, tipos, estadísticas descriptivas, distribución de clases, histogramas y boxplots por clase, valores faltantes, duplicados y matriz de correlación.
3. **Limpieza de datos:** eliminación de duplicados exactos e imputación de `bare_nuclei` con la mediana de cada clase, con gráficas comparativas antes y después.
4. **Aumento de datos:** SMOTE para igualar las clases, proyección con PCA para ubicar los registros sintéticos y comparación de boxplots y correlaciones antes y después.
5. **Conclusiones y referencias.**

## Resultados principales

- El dataset original tiene 699 registros: 458 benignos (65.52 %) y 241 malignos (34.48 %).
- Se encontraron 8 duplicados exactos; tras eliminarlos quedan 691 registros (453 benignos y 238 malignos).
- Hay 16 valores faltantes (2.29 %), todos en `bare_nuclei`: 14 en casos benignos y 2 en malignos. Se imputaron con la mediana de su clase (1 y 10, respectivamente), con cambios mínimos en la desviación estándar y en las correlaciones.
- Los atributos más correlacionados con el diagnóstico son `bare_nuclei`, `cell_shape` y `cell_size` (r ≈ 0.82). `mitoses` es el más débil (r ≈ 0.42). `cell_size` y `cell_shape` son muy redundantes entre sí (r ≈ 0.91).
- SMOTE generó 215 registros sintéticos y dejó ambas clases con 453 casos. Las correlaciones de los atributos con la clase bajaron ligeramente después del aumento.

## Requisitos

- Python 3.9 o superior
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` (PCA)
- `imbalanced-learn` (SMOTE)

Instalación:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Cómo ejecutarlo

### En Google Colab
1. Abre el notebook en Colab.
2. Si hace falta, ejecuta `!pip install imbalanced-learn` en una celda.
3. Ejecuta todas las celdas en orden (`Entorno de ejecución > Ejecutar todo`).

### En local

1. Clona el repositorio y entra a la carpeta del notebook.
2. Instala las dependencias.
3. Abre `Notebook_preprocesamiento.ipynb` con Jupyter y ejecuta las celdas en orden.

El notebook carga los datos directamente desde GitHub con `pd.read_csv`, por lo que **requiere conexión a internet**:

```python
url = "https://github.com/mareylugo/ICD-2026/raw/refs/heads/main/Tareas/Notebook%20preprocesamiento/breast-cancer-wisconsin.data"
df = pd.read_csv(url, header=None, names=cols, na_values="?")
```

Si prefieres trabajar sin conexión, descarga `breast-cancer-wisconsin.data` y reemplaza `url` por la ruta local del archivo.

## Notas importantes

- **Orden de ejecución:** algunas celdas (porcentaje de faltantes, registros con `bare_nuclei` vacío) solo dan resultados correctos **antes** de la imputación. Ejecuta el notebook de arriba hacia abajo.
- **Fuga de información:** SMOTE y la imputación por clase se aplicaron al conjunto completo con fines exploratorios. Para entrenar un modelo deben aplicarse solo al conjunto de entrenamiento, después de dividir los datos.
- **IDs repetidos:** varios `id` aparecen más de una vez con valores distintos (posibles muestras distintas del mismo paciente). No se eliminaron, pero conviene agruparlos por `id` al dividir en entrenamiento y prueba.

## Trabajo futuro

- Extracción de características y reducción de dimensionalidad (por ejemplo, combinar `cell_size` y `cell_shape`, o evaluar PCA).
- Evaluar si `mitoses` aporta información con un modelo antes de descartarla.
- Entrenar y comparar clasificadores con y sin SMOTE, usando validación cruzada.

## Referencias


[1] W. H. Wolberg and O. L. Mangasarian, "Breast Cancer Wisconsin (Original)," UCI Machine Learning Repository, 1992. [Online]. Available: https://doi.org/10.24432/C5HP4Z

[2] W. H. Wolberg and O. L. Mangasarian, "Multisurface method of pattern separation for medical diagnosis applied to breast cytology," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 87, pp. 9193–9196, Dec. 1990.

[3] O. L. Mangasarian and W. H. Wolberg, "Cancer diagnosis via linear programming," *SIAM News*, vol. 23, no. 5, pp. 1, 18, Sep. 1990.

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.
