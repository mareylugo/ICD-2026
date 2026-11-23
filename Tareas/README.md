# Tarea · CNN compacta con Global Average Pooling

**Introducción a la Ciencia de Datos 2026** · Posgrado en Ciencias de la Computación, CICESE

Esta tarea diseña una CNN con *Global Average Pooling* (GAP) que supere el **95 % de exactitud de validación** en MNIST con **menos de 54 330 parámetros entrenables**, que es el total de la CNN base con Flatten. Después mide cómo cambia su exactitud cuando los dígitos se desplazan 0, 2 y 4 píxeles.

## Contenido

| Archivo | Descripción |
|---|---|
| `tarea_cnn_gap.ipynb` | Notebook ejecutado: modelos de referencia, 11 variantes GAP, comparaciones, selección, evaluación con desplazamiento y conclusiones. |
| `README.md` | Este archivo. |

## Configuración

Es la misma del notebook de clase `icd-21-cnn.ipynb`:

- **Datos:** subconjuntos estratificados de MNIST (8000 de entrenamiento, 1500 de validación y 2000 de prueba) con semilla 42. Cada imagen lleva un margen negro de 4 px y queda de 36 × 36 × 1.
- **Entrenamiento:** Adam con tasa 0.001, lote 128 y 20 épocas, con la misma semilla en todos los modelos.
- **Selección:** se usa la exactitud de validación en la época 20. El conjunto de prueba solo se usa en la comparación final con desplazamiento.

## Arquitecturas

Cada variante es una secuencia de etapas. Una etapa tiene una o más convoluciones de 3 × 3 con ReLU y *padding same*, seguidas de un *max pooling* de 2 × 2. Después de la última etapa van GAP y una capa densa de 10 clases.

La notación `16 | 32` indica dos etapas, con 16 y 32 filtros. `16-16` indica dos convoluciones seguidas dentro de una misma etapa.

| Variante | Arquitectura | Campo receptivo (px) | Parámetros | Validación (%) |
|---|---|---:|---:|---:|
| CNN + Flatten | `16-16` + Flatten | 6 | 54 330 | 96.80 |
| GAP base | `16-16` | 6 | 2 650 | 43.93 |
| V9 | `8 \| 8 \| 8` | 22 | 1 338 | 81.13 |
| V10 | `8 \| 8 \| 16` | 22 | 2 002 | 89.20 |
| V1 | `16 \| 16` | 10 | 2 650 | 61.80 |
| V6 | `8 \| 16 \| 16` | 22 | 3 738 | 91.53 |
| V4 | `16 \| 16 \| 16` | 22 | 4 970 | 90.07 |
| V7 | `16 \| 16 \| 16`, kernel 5 × 5 en la primera convolución | 24 | 5 226 | 90.47 |
| V2 | `16-16-16-16` | 10 | 7 290 | 61.20 |
| V3 | `16-16 \| 16-16` | 16 | 7 290 | 88.33 |
| **V5** | **`16 \| 16 \| 16 \| 16`** | **46** | **7 290** | **95.53** |
| V8 | `16 \| 32 \| 32` | 22 | 14 378 | 93.47 |
| V11 | `16 \| 32 \| 64` | 22 | 23 946 | 95.33 |

Solo V5 y V11 superan el 95 %. Se seleccionó **V5** porque tiene menos parámetros.

### Comparaciones de una sola decisión

| Comparación | Cambio | Δ validación (pp) | Δ parámetros |
|---|---|---:|---:|
| C1 | Pooling entre convoluciones (GAP base → V1) | +17.87 | 0 |
| C2 | Pooling entre convoluciones (V2 → V3) | +27.13 | 0 |
| C3a | Tercera etapa (V1 → V4) | +28.27 | +2 320 |
| C3b | Cuarta etapa (V4 → V5) | +5.47 | +2 320 |
| C4 | Primera convolución de 8 a 16 filtros (V6 → V4) | −1.47 | +1 232 |
| C5 | Kernel de la primera convolución de 3 × 3 a 5 × 5 (V4 → V7) | +0.40 | +256 |
| C6 | Etapas profundas de 16 a 32 filtros (V4 → V8) | +3.40 | +9 408 |

## Efecto del desplazamiento (prueba)

Para 2 y 4 px se promedian las cuatro direcciones: arriba, abajo, izquierda y derecha. No se reentrena ningún modelo.

| Modelo | Parámetros | 0 px (%) | 2 px (%) | 4 px (%) | Caída Acc(0px) − Acc(4px) (pp) |
|---|---:|---:|---:|---:|---:|
| CNN + Flatten | 54 330 | 97.15 | 82.81 | 37.36 | 59.79 |
| CNN + GAP base | 2 650 | 46.20 | 46.19 | 46.22 | −0.02 |
| GAP seleccionado (V5) | 7 290 | 96.05 | 89.22 | 67.79 | 28.26 |

## Conclusiones

- **Cambio más importante.** Fue colocar un *max pooling* después de cada convolución, lo que amplía el campo receptivo de 6 a 46 px antes de GAP. V2 y V5 tienen los mismos 7 290 parámetros y solo difieren en la ubicación del pooling, y la exactitud de validación pasa de 61.20 % a 95.53 %. En cambio, ensanchar la red rindió poco: V11 tiene el triple de parámetros y queda por debajo de V5.
- **Ahorro de parámetros.** V5 ahorra **47 040 parámetros frente a Flatten (86.6 %, 7.5 veces menos)**, a cambio de 1.27 pp menos de validación.
- **Desplazamiento.** Centrados, Flatten y V5 quedan a 1.10 pp. Desde **2 px** V5 es claramente mejor, con +6.41 pp, y a 4 px la ventaja llega a +30.42 pp. El GAP base casi no cae, pero se queda en 46 % en todas las distancias. Por eso hay que mirar tanto la caída como la exactitud que se conserva.
- **Limitaciones.** V5 supera el umbral por solo 0.53 pp, unas 8 imágenes de validación, en una sola ejecución con una semilla. Además, V5 no es invariante a la traslación: pierde 28 pp a 4 px por el paso del pooling y los efectos de borde.

## Cómo ejecutar

Abre `tarea_cnn_gap.ipynb` en Google Colab o Jupyter y ejecuta las celdas en orden. MNIST se descarga en la primera ejecución. En CPU de Colab, el entrenamiento de los 13 modelos tardó unos 42 minutos.

```python
%pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Declaración de uso de IA

Se utilizó el asistente de inteligencia artificial Claude (Anthropic) en la elaboración de este notebook. La autora revisó el contenido y es responsable de su exactitud.
