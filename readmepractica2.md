# Práctica 2: Procesamiento de datasets

**Curso:** Introducción a la Ciencia de Datos 2026
**Programa:** Posgrado en Ciencias de la Computación, CICESE
**Autora:** Luz Marey Lugo Acedo
**Entregable:** `Practica2.ipynb` (Google Colab)

## Descripción

Preprocesamiento del dataset *Cervical Cancer Risk Factors* (UCI Machine Learning Repository), un censo de factores de riesgo de cáncer cervical del Hospital Universitario de Caracas con 858 registros y 36 atributos. Los valores faltantes están codificados con `?`.

Técnicas aplicadas en el notebook:

1. Exploración y limpieza de datos (faltantes, atributos casi vacíos y registros duplicados).
2. Aumento de datos (SMOTENC).
3. Extracción de características.
4. Reducción de dimensionalidad (PCA).
5. Selección de características.

## Cómo ejecutarlo

1. Abrir `Practica2.ipynb` en Google Colab.
2. Ejecutar las celdas en orden. El dataset se descarga desde el repositorio de GitHub indicado en la primera celda de código.
3. La sección de aumento de datos usa `imbalanced-learn`, que Colab suele traer instalada. Si hay error de importación, ejecutar `!pip install imbalanced-learn`.

## Declaración de uso de inteligencia artificial

Se utilizó el asistente de inteligencia artificial Claude (Anthropic) en la elaboración del código en este notebook. La autora revisó el contenido y es responsable de su exactitud.
