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

## Referencias

[1] Organización Mundial de la Salud, «Cáncer de cuello uterino», Notas descriptivas de la OMS, 3 de julio de 2026. [En línea]. Disponible en: <https://www.who.int/es/news-room/fact-sheets/detail/cervical-cancer>. [Accedido: 17-sep-2026].

[2] K. Fernandes, J. Cardoso, and J. Fernandes. "Cervical Cancer (Risk Factors)," UCI Machine Learning Repository, 2017. [Online]. Available: <https://doi.org/10.24432/C5Z310>.

[3] M. Likhite, "Cervical Cancer Diagnosis with Machine Learning: Part 1 — Data," Data Science Collective, Medium, Oct. 19, 2025. [Online]. Available: <https://medium.com/data-science-collective/cervical-cancer-diagnosis-with-machine-learning-part-1-data-f1e17af779f5>. [Accessed: Sep. 17, 2026].

[4] "La madre más prolífica de la historia," Guinness World Records. [Online]. Available: <https://www.guinnessworldrecords.com/world-records/most-prolific-mother-ever>. [Accessed: Sep. 18, 2026].

[5] "La persona más longeva de la historia," Guinness World Records. [Online]. Available: <https://www.guinnessworldrecords.com/world-records/oldest-person>. [Accessed: Sep. 18, 2026].
