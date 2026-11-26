"""Prepara el subconjunto de señales crudas de FNUSA para entrenar la CNN 1D en Colab.

Uso (desde la raíz del repositorio, después de crear_pliegues.py):
    python scripts/preparar_subconjunto_cnn.py --zip datos/crudos/Dataset_Fnusa.zip

Salida:
    datos/cnn_subconjunto_fnusa.npz       x (n, 3750) float16, y, segment_id, patient_id, pliegue
    datos/cnn_subconjunto_ids.csv         segment_id y pliegue (lo usa el notebook principal)

Decisiones:
- Se toma ~21 % de cada combinación pliegue × clase (~40 mil fragmentos), con semilla fija:
  conserva la proporción de clases y los pliegues por paciente de datos/pliegues.csv.
- Se submuestrea de 5 kHz a 1250 Hz (factor 4) con un filtro paso bajas en frecuencia
  (corte en 600 Hz, por debajo de la nueva frecuencia de Nyquist de 625 Hz). Se conserva la
  banda de fast ripple (250–500 Hz) y el archivo baja a ~300 MB, manejable en Colab.
"""

import argparse
import time

import numpy as np

from lectura_mat import LectorZip

SEMILLA = 42
FRACCION = 0.21
FS = 5000
FACTOR = 4                  # 5000 Hz -> 1250 Hz
CORTE_HZ = 600


def submuestrear(x):
    """Filtro paso bajas ideal en frecuencia y submuestreo por FACTOR. x: (B, 15000)."""
    espectro = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(x.shape[1], 1 / FS)
    espectro[:, f > CORTE_HZ] = 0
    filtrada = np.fft.irfft(espectro, n=x.shape[1], axis=1)
    return filtrada[:, ::FACTOR]


def main(ruta_zip):
    # Lectura de pliegues sin pandas, para que el script solo dependa de NumPy
    datos = np.genfromtxt("datos/pliegues.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    datos = datos[datos["hospital"] == "fnusa"]

    rng = np.random.default_rng(SEMILLA)
    elegidos = []
    for k in np.unique(datos["pliegue"]):
        for clase in np.unique(datos["category_name"]):
            idx = np.flatnonzero((datos["pliegue"] == k) & (datos["category_name"] == clase))
            if len(idx):
                n = max(1, int(round(FRACCION * len(idx))))
                elegidos.extend(rng.choice(idx, size=n, replace=False))
    sub = datos[np.sort(np.array(elegidos))]
    print(f"{len(sub)} fragmentos seleccionados")

    lector = LectorZip(ruta_zip)
    x = np.empty((len(sub), 15000 // FACTOR), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(sub), 256):
        lote = np.stack([lector.leer(s) for s in sub["segment_id"][i:i + 256]]).astype(np.float64)
        x[i:i + len(lote)] = submuestrear(lote).astype(np.float16)
        if (i // 256) % 40 == 0:
            print(f"{i + len(lote):>6} / {len(sub)}  ({time.time() - t0:.0f} s)", flush=True)

    np.savez("datos/cnn_subconjunto_fnusa.npz", x=x, y=sub["category_name"].astype(str),
             segment_id=sub["segment_id"].astype(str), patient_id=sub["patient_id"],
             pliegue=sub["pliegue"])
    with open("datos/cnn_subconjunto_ids.csv", "w", encoding="utf-8") as fh:
        fh.write("segment_id,pliegue\n")
        for s, k in zip(sub["segment_id"], sub["pliegue"]):
            fh.write(f"{s},{k}\n")
    print(f"Listo en {time.time() - t0:.0f} s: datos/cnn_subconjunto_fnusa.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="datos/crudos/Dataset_Fnusa.zip")
    main(p.parse_args().zip)
