"""Extracción de atributos espectrales, temporales y de eventos de cada fragmento de iEEG.

Uso (desde la raíz del repositorio):
    python scripts/extraer_atributos.py --hospital fnusa --zip datos/crudos/Dataset_Fnusa.zip
    python scripts/extraer_atributos.py --hospital mayo  --zip datos/crudos/Dataset_Mayo.zip

Salida:
    datos/segments_<hospital>.csv          metadatos y etiquetas (copiados del zip)
    datos/atributos_<hospital>.csv.gz      una fila por fragmento: segment_id + atributos

Cada fragmento dura 3 s a 5 kHz (15000 muestras) y viene normalizado (desv. estándar 1),
por eso las potencias se expresan como proporción de la potencia total. El cálculo es
independiente por fragmento: no usa estadísticas de otros fragmentos, así que no puede
filtrar información entre entrenamiento y prueba.

Solo depende de NumPy. Para correr por partes (p. ej., en una máquina con poco tiempo
por proceso) se usan --inicio y --n, y después --unir junta las partes.
"""

import argparse
import csv
import gzip
import hashlib
import io
import time
from pathlib import Path

import numpy as np

from lectura_mat import LectorZip

FS = 5000                     # frecuencia de muestreo (Hz)
N_MUESTRAS = 15000            # 3 s
FRECUENCIA_RED = {"fnusa": 50, "mayo": 60}   # Brno (Europa) y Rochester (EE. UU.)

# Bandas de potencia relativa (Hz), límite inferior incluido y superior excluido
BANDAS = {
    "delta": (1, 4), "theta": (4, 8), "alfa": (8, 13), "beta": (13, 30),
    "gamma": (30, 80), "ripple": (80, 250), "fast_ripple": (250, 500),
    "muy_alta": (500, 1000),
}
RANGO_TOTAL = (1, 1000)       # potencia de referencia para las proporciones
WELCH_NPERSEG = 5000          # ventanas de 1 s -> resolución de 1 Hz
WELCH_PASO = 2500             # 50 % de traslape -> 5 ventanas por fragmento
BORDE_ENVOLVENTE = 250        # 50 ms que se ignoran en cada extremo (efecto de borde)
EPS = 1e-12                   # evita divisiones y logaritmos de cero

COLUMNAS = (
    ["segment_id", "md5", "media", "desv_est", "pot_rel_lt1"]
    + [f"pot_rel_{b}" for b in BANDAS]
    + ["pot_rel_red", "cociente_ripple_gamma", "cociente_fr_ripple",
       "frec_pico", "frec_mediana", "borde_espectral_95", "entropia_espectral",
       "planitud_espectral", "pendiente_espectral",
       "line_length", "curtosis", "asimetria", "max_abs", "pico_a_pico",
       "cruces_cero_por_s", "max_dif", "hjorth_movilidad", "hjorth_complejidad",
       "n_picos_3de", "env_max_ripple", "env_max_fast_ripple"]
)


def welch_lote(x):
    """Densidad espectral por el método de Welch para un lote (B, N). Devuelve (f, P)."""
    ventana = np.hanning(WELCH_NPERSEG)
    inicios = range(0, x.shape[1] - WELCH_NPERSEG + 1, WELCH_PASO)
    potencia = 0.0
    for i in inicios:
        seg = x[:, i:i + WELCH_NPERSEG]
        seg = seg - seg.mean(axis=1, keepdims=True)          # quitar tendencia constante
        potencia = potencia + np.abs(np.fft.rfft(seg * ventana, axis=1)) ** 2
    f = np.fft.rfftfreq(WELCH_NPERSEG, 1 / FS)
    return f, potencia / len(inicios)


def envolvente_max(espectro, f, banda):
    """Máximo de la envolvente (amplitud de Hilbert) de la señal filtrada en una banda.

    Se filtra en frecuencia y se construye la señal analítica en el mismo paso:
    se conservan solo las frecuencias positivas de la banda, multiplicadas por 2.
    """
    analitica = np.zeros((espectro.shape[0], N_MUESTRAS), dtype=np.complex128)
    en_banda = (f >= banda[0]) & (f < banda[1])
    analitica[:, :len(f)][:, en_banda] = 2 * espectro[:, en_banda]
    envolvente = np.abs(np.fft.ifft(analitica, axis=1))
    return envolvente[:, BORDE_ENVOLVENTE:-BORDE_ENVOLVENTE].max(axis=1)


def atributos_lote(x, hospital):
    """Calcula todos los atributos para un lote de señales x de forma (B, 15000)."""
    x = x.astype(np.float64)
    media = x.mean(axis=1)
    desv = x.std(axis=1)
    xc = x - media[:, None]
    res = {"media": media, "desv_est": desv}

    # --- Espectrales (Welch, resolución 1 Hz) ---
    f, P = welch_lote(xc)
    en_total = (f >= RANGO_TOTAL[0]) & (f < RANGO_TOTAL[1])
    total = P[:, en_total].sum(axis=1) + EPS

    def pot(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(axis=1)

    for nombre, (lo, hi) in BANDAS.items():
        res[f"pot_rel_{nombre}"] = pot(lo, hi) / total
    red = FRECUENCIA_RED[hospital]
    res["pot_rel_red"] = pot(red - 2, red + 3) / total          # red ± 2 Hz
    res["cociente_ripple_gamma"] = pot(80, 250) / (pot(30, 80) + EPS)
    res["cociente_fr_ripple"] = pot(250, 500) / (pot(80, 250) + EPS)

    Pt, ft = P[:, en_total], f[en_total]
    p = Pt / Pt.sum(axis=1, keepdims=True)                        # espectro como distribución
    acumulada = np.cumsum(p, axis=1)
    res["frec_pico"] = ft[Pt.argmax(axis=1)]
    res["frec_mediana"] = ft[(acumulada < 0.5).sum(axis=1)]
    res["borde_espectral_95"] = ft[np.minimum((acumulada < 0.95).sum(axis=1), len(ft) - 1)]
    res["entropia_espectral"] = -(p * np.log(p + EPS)).sum(axis=1) / np.log(p.shape[1])
    res["planitud_espectral"] = np.exp(np.log(Pt + EPS).mean(axis=1)) / (Pt.mean(axis=1) + EPS)

    # Pendiente 1/f: recta de mínimos cuadrados en escala log-log entre 2 y 200 Hz,
    # sin la frecuencia de red (± 3 Hz) para que el ruido de línea no la sesgue
    en_pend = (f >= 2) & (f < 200) & (np.abs(f - red) > 3)
    lx = np.log10(f[en_pend])
    ly = np.log10(P[:, en_pend] + EPS)
    lx_c = lx - lx.mean()
    res["pendiente_espectral"] = ((ly - ly.mean(axis=1, keepdims=True)) * lx_c).sum(axis=1) / (lx_c ** 2).sum()

    # Potencia por debajo de 1 Hz: Welch con ventanas de 1 s no la resuelve, así que se
    # usa el periodograma del fragmento completo (resolución de 1/3 Hz)
    espectro = np.fft.rfft(xc, axis=1)
    f_comp = np.fft.rfftfreq(N_MUESTRAS, 1 / FS)
    P_comp = np.abs(np.fft.rfft(xc * np.hanning(N_MUESTRAS), axis=1)) ** 2
    bajo = (f_comp > 0) & (f_comp <= 1)
    tot_comp = P_comp[:, (f_comp > 0) & (f_comp < RANGO_TOTAL[1])].sum(axis=1) + EPS
    res["pot_rel_lt1"] = P_comp[:, bajo].sum(axis=1) / tot_comp

    # --- Temporales ---
    dif = np.diff(xc, axis=1)
    dif2 = np.diff(dif, axis=1)
    var = xc.var(axis=1) + EPS
    res["line_length"] = np.abs(dif).mean(axis=1)
    res["curtosis"] = (xc ** 4).mean(axis=1) / var ** 2 - 3        # exceso de curtosis
    res["asimetria"] = (xc ** 3).mean(axis=1) / var ** 1.5
    res["max_abs"] = np.abs(xc).max(axis=1)
    res["pico_a_pico"] = xc.max(axis=1) - xc.min(axis=1)
    signo = np.signbit(xc)
    res["cruces_cero_por_s"] = (signo[:, 1:] != signo[:, :-1]).sum(axis=1) / (N_MUESTRAS / FS)
    res["max_dif"] = np.abs(dif).max(axis=1)
    movilidad = np.sqrt(dif.var(axis=1) / var)
    res["hjorth_movilidad"] = movilidad
    res["hjorth_complejidad"] = np.sqrt(dif2.var(axis=1) / (dif.var(axis=1) + EPS)) / (movilidad + EPS)
    sobre = np.abs(xc) > 3 * np.sqrt(var)[:, None]                  # excursiones > 3 desv. est.
    res["n_picos_3de"] = (sobre[:, 1:] & ~sobre[:, :-1]).sum(axis=1)

    # --- Eventos: ráfagas de alta frecuencia ---
    res["env_max_ripple"] = envolvente_max(espectro, f_comp, BANDAS["ripple"])
    res["env_max_fast_ripple"] = envolvente_max(espectro, f_comp, BANDAS["fast_ripple"])
    return res


def procesar(args):
    """Calcula los atributos de los fragmentos [inicio, inicio + n) y los escribe."""
    lector = LectorZip(args.zip)
    datos = Path("datos")
    datos.mkdir(exist_ok=True)
    (datos / f"segments_{args.hospital}.csv").write_bytes(lector.segments_csv())

    ids = lector.ids[args.inicio:args.inicio + args.n] if args.n else lector.ids[args.inicio:]
    if args.n:
        partes = datos / "partes"
        partes.mkdir(exist_ok=True)
        salida = partes / f"atributos_{args.hospital}_{args.inicio:06d}.csv.gz"
    else:
        salida = datos / f"atributos_{args.hospital}.csv.gz"

    t0 = time.time()
    with gzip.open(salida, "wt", newline="") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(COLUMNAS)
        for i in range(0, len(ids), args.lote):
            lote_ids = ids[i:i + args.lote]
            senales = [lector.leer(s) for s in lote_ids]
            huellas = [hashlib.md5(s.tobytes()).hexdigest() for s in senales]   # detecta duplicados
            res = atributos_lote(np.stack(senales), args.hospital)
            for k, sid in enumerate(lote_ids):
                fila = [sid, huellas[k]] + [f"{res[c][k]:.6g}" for c in COLUMNAS[2:]]
                escritor.writerow(fila)
            if (i // args.lote) % 40 == 0:
                print(f"{i + len(lote_ids):>7} / {len(ids)} fragmentos  ({time.time() - t0:.0f} s)", flush=True)
    print(f"Listo: {salida} ({len(ids)} fragmentos, {time.time() - t0:.0f} s)")


def unir(hospital):
    """Junta las partes de un hospital en datos/atributos_<hospital>.csv.gz."""
    partes = sorted(Path("datos/partes").glob(f"atributos_{hospital}_*.csv.gz"))
    salida = Path(f"datos/atributos_{hospital}.csv.gz")
    n = 0
    with gzip.open(salida, "wt", newline="") as out:
        for j, parte in enumerate(partes):
            with gzip.open(parte, "rt") as fh:
                encabezado = fh.readline()
                if j == 0:
                    out.write(encabezado)
                for linea in fh:
                    out.write(linea)
                    n += 1
    print(f"Listo: {salida} ({n} fragmentos de {len(partes)} partes)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hospital", choices=["fnusa", "mayo"], required=True)
    p.add_argument("--zip", help="ruta al Dataset_Fnusa.zip o Dataset_Mayo.zip")
    p.add_argument("--inicio", type=int, default=0, help="primer fragmento (orden alfabético)")
    p.add_argument("--n", type=int, default=0, help="número de fragmentos; 0 = todos")
    p.add_argument("--lote", type=int, default=128, help="fragmentos por lote vectorizado")
    p.add_argument("--unir", action="store_true", help="solo juntar las partes ya calculadas")
    a = p.parse_args()
    if a.unir:
        unir(a.hospital)
    else:
        procesar(a)
