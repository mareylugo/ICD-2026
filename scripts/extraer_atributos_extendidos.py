"""Atributos adicionales con librerías especializadas, sobre el subconjunto de la CNN.

Uso (desde la raíz del repositorio, después de preparar_subconjunto_cnn.py):
    python scripts/extraer_atributos_extendidos.py                 # todo el subconjunto
    python scripts/extraer_atributos_extendidos.py --limite 200    # prueba rápida
    python scripts/extraer_atributos_extendidos.py --unir          # solo juntar las partes

Entrada:
    datos/cnn_subconjunto_fnusa.npz    40,539 fragmentos de FNUSA a 1250 Hz (ver preparar_subconjunto_cnn.py)

Salida:
    datos/partes_extendidos/parte_<inicio>.csv.gz      partes intermedias (permiten retomar)
    datos/atributos_extendidos_subconjunto.csv.gz      una fila por fragmento: segment_id + atributos

Atributos por librería (prefijo de columna):
    sp_     specparam: componente aperiódica (offset, codo, exponente) y picos del espectro.
    ant_    antropy: entropías (permutación, espectral, SVD, muestral, aproximada) y
            dimensiones fractales (Petrosian, Katz, Higuchi, DFA).
    mne_    MNE: potencia relativa por banda con el método multitaper.
    epy_    EPyCom: detectores de HFO (line length, RMS, Hilbert) y de espigas (Barkmeier).
    tsfel_  TSFEL: dominios estadístico, temporal y espectral (el fractal se excluye: tarda
            unos 7 s por fragmento, ~80 h para el subconjunto).

Como en extraer_atributos.py, cada fragmento se procesa por separado y sin estadísticas de
otros fragmentos, así que el cálculo no filtra información entre entrenamiento y prueba.
Las señales vienen normalizadas (desv. est. 1), por eso las potencias son relativas.
"""

import os

# Un hilo por proceso: el paralelismo se hace entre procesos (antes de importar NumPy)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

FS = 1250                         # frecuencia del subconjunto (submuestreado de 5 kHz)
FRECUENCIA_RED = 50               # FNUSA (Brno)
BORDE_S = 0.1                     # eventos a menos de 100 ms de un extremo se ignoran (efecto del filtro)
VENTANA_HFO = int(0.02 * FS)      # 20 ms: equivale a las 100 muestras por omisión de EPyCom a 5 kHz
BANDAS = {
    "delta": (1, 4), "theta": (4, 8), "alfa": (8, 13), "beta": (13, 30),
    "gamma": (30, 80), "ripple": (80, 250), "fast_ripple": (250, 500),
    "sobre_500": (500, 600),      # el subconjunto se filtró por debajo de 600 Hz
}
RANGO_TOTAL = (1, 600)
EPS = 1e-12

ENTRADA = Path("datos/cnn_subconjunto_fnusa.npz")
PARTES = Path("datos/partes_extendidos")
SALIDA = Path("datos/atributos_extendidos_subconjunto.csv.gz")


# ---------------------------------------------------------------- por librería
def atr_specparam(s):
    """Modelo espectral: aperiódico con codo (knee) + hasta 6 picos gaussianos, 1–500 Hz."""
    from scipy.signal import welch
    from specparam import SpectralModel

    f, p = welch(s, fs=FS, nperseg=FS // 2)                       # resolución de 2 Hz
    m = SpectralModel(aperiodic_mode="knee", max_n_peaks=6, verbose=False)
    m.fit(f, p, [1, 500])
    offset, knee, exponente = m.results.get_params("aperiodic")
    picos = np.atleast_2d(m.results.get_params("peak"))           # filas: (frecuencia, potencia, ancho)
    picos = picos[~np.isnan(picos).any(axis=1)] if picos.size else np.empty((0, 3))
    metricas = m.results.metrics.results

    def pico_en(lo, hi):
        en = (picos[:, 0] >= lo) & (picos[:, 0] < hi)
        return picos[en, 1].max() if en.any() else 0.0

    mayor = picos[picos[:, 1].argmax()] if len(picos) else (np.nan, 0.0, np.nan)
    return {
        "sp_offset": offset, "sp_knee": knee, "sp_exponente": exponente,
        "sp_n_picos": len(picos), "sp_pico_mayor_frec": mayor[0], "sp_pico_mayor_pot": mayor[1],
        "sp_pico_ripple_pot": pico_en(80, 250), "sp_pico_fr_pot": pico_en(250, 500),
        "sp_r2": metricas.get("gof_rsquared", np.nan), "sp_error": metricas.get("error_mae", np.nan),
    }


def atr_antropy(s):
    """Entropías y dimensiones fractales de antropy."""
    import antropy as ant

    return {
        "ant_perm_entropia": ant.perm_entropy(s, normalize=True),
        "ant_entropia_espectral": ant.spectral_entropy(s, sf=FS, method="welch", normalize=True),
        "ant_svd_entropia": ant.svd_entropy(s, normalize=True),
        "ant_entropia_muestral": ant.sample_entropy(s),
        "ant_entropia_aproximada": ant.app_entropy(s),
        "ant_petrosian_fd": ant.petrosian_fd(s),
        "ant_katz_fd": ant.katz_fd(s),
        "ant_higuchi_fd": ant.higuchi_fd(s),
        "ant_dfa": ant.detrended_fluctuation(s),
    }


def atr_mne(s):
    """Potencia relativa por banda con el estimador multitaper de MNE."""
    from mne.time_frequency import psd_array_multitaper

    p, f = psd_array_multitaper(s, sfreq=FS, fmin=RANGO_TOTAL[0], fmax=RANGO_TOTAL[1], verbose=False)
    total = p.sum() + EPS
    res = {f"mne_pot_rel_{b}": p[(f >= lo) & (f < hi)].sum() / total for b, (lo, hi) in BANDAS.items()}
    res["mne_pot_rel_red"] = p[np.abs(f - FRECUENCIA_RED) <= 2].sum() / total
    return res


def atr_epycom(s):
    """Detectores de EPyCom: número y duración total de eventos por fragmento."""
    from epycom.event_detection.hfo.hilbert_detector import detect_hfo_hilbert
    from epycom.event_detection.hfo.ll_detector import detect_hfo_ll
    from epycom.event_detection.hfo.rms_detector import detect_hfo_rms
    from epycom.event_detection.spike.barkmeier_detector import detect_spikes_barkmeier
    from scipy.signal import butter, sosfiltfilt

    margen = BORDE_S * FS
    n = len(s)

    def validos(eventos):
        """Eventos (inicio, fin, ...) que no tocan los bordes del fragmento."""
        return [e for e in eventos if e[0] >= margen and e[1] <= n - margen]

    res = {}
    for banda, (lo, hi) in (("ripple", (80, 250)), ("fr", (250, 500))):
        filtrada = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), s)
        for nombre, detector in (("ll", detect_hfo_ll), ("rms", detect_hfo_rms)):
            ev = validos(detector(filtrada, fs=FS, window_size=VENTANA_HFO))
            res[f"epy_{nombre}_{banda}_n"] = len(ev)
            res[f"epy_{nombre}_{banda}_dur_s"] = sum(e[1] - e[0] for e in ev) / FS
    # Hilbert: (inicio, fin, frec. mínima, frec. máxima, frec. pico, amplitud)
    ev = validos(detect_hfo_hilbert(s, FS, 80, 500, num_bands=50))
    res["epy_hilbert_n"] = len(ev)
    res["epy_hilbert_amp_max"] = max((e[5] for e in ev), default=0.0)
    res["epy_hilbert_frec_media"] = float(np.mean([e[4] for e in ev])) if ev else 0.0
    # Barkmeier: una tupla por espiga, la primera entrada es su posición (muestra)
    espigas = [e for e in detect_spikes_barkmeier(s, fs=FS) if margen <= e[0] <= n - margen]
    res["epy_espigas_n"] = len(espigas)
    return res


_TSFEL_CFG = None


def atr_tsfel(s):
    """TSFEL: dominios estadístico, temporal y espectral (sin el fractal)."""
    global _TSFEL_CFG
    import tsfel

    if _TSFEL_CFG is None:
        _TSFEL_CFG = {d: tsfel.get_features_by_domain(d)[d] for d in ("statistical", "temporal", "spectral")}
    df = tsfel.time_series_features_extractor(_TSFEL_CFG, s, fs=FS, verbose=0)
    return {"tsfel_" + c.replace("0_", "", 1).replace(" ", "_").lower(): float(v)
            for c, v in df.iloc[0].items()}


LIBRERIAS = {"specparam": atr_specparam, "antropy": atr_antropy, "mne": atr_mne,
             "epycom": atr_epycom, "tsfel": atr_tsfel}


def atributos_fragmento(s):
    """Todos los atributos de un fragmento. Si una librería falla, sus columnas quedan vacías."""
    fila = {}
    for nombre, funcion in LIBRERIAS.items():
        try:
            fila.update(funcion(s))
        except Exception as e:                       # se registra y se sigue con las demás
            fila[f"error_{nombre}"] = f"{type(e).__name__}: {e}"[:200]
    return fila


def procesar_parte(inicio, x, ids, ruta):
    """Calcula una parte y la guarda; se ejecuta en un proceso aparte."""
    warnings.filterwarnings("ignore")
    filas = []
    for sid, s in zip(ids, x):
        fila = {"segment_id": sid}
        fila.update(atributos_fragmento(s.astype(np.float64)))
        filas.append(fila)
    pd.DataFrame(filas).to_csv(ruta, index=False, compression="gzip")
    return inicio, len(filas)


def unir(salida=SALIDA, carpeta=PARTES):
    """Junta las partes en un solo archivo y reporta errores por librería."""
    partes = sorted(carpeta.glob("parte_*.csv.gz"))
    df = pd.concat([pd.read_csv(p) for p in partes], ignore_index=True)
    errores = [c for c in df.columns if c.startswith("error_")]
    for c in errores:
        print(f"  {c}: {df[c].notna().sum()} fragmentos con error; ej.: {df[c].dropna().iloc[0]}")
    df.drop(columns=errores).to_csv(salida, index=False, compression="gzip")
    print(f"Listo: {salida} ({len(df):,} fragmentos, {df.shape[1] - len(errores) - 1} atributos, "
          f"{len(partes)} partes)")


def main(args):
    datos = np.load(ENTRADA)
    x, ids = datos["x"], datos["segment_id"]
    if args.limite:                                  # prueba: muestra aleatoria con semilla fija
        elegidos = np.sort(np.random.default_rng(42).choice(len(ids), args.limite, replace=False))
        x, ids = x[elegidos], ids[elegidos]
    # Las pruebas van a otra carpeta, para no confundirse con las partes de la corrida completa
    carpeta = PARTES.with_name(PARTES.name + "_prueba") if args.limite else PARTES
    carpeta.mkdir(parents=True, exist_ok=True)
    if args.limite:
        for p in carpeta.glob("parte_*.csv.gz"):
            p.unlink()

    pendientes = []
    for inicio in range(0, len(ids), args.parte):
        ruta = carpeta / f"parte_{inicio:06d}.csv.gz"
        if not ruta.exists():                        # permite retomar una corrida interrumpida
            pendientes.append((inicio, ruta))
    print(f"{len(ids):,} fragmentos, {len(pendientes)} partes pendientes de {args.parte}, "
          f"{args.procesos} procesos", flush=True)

    t0 = time.time()
    hechos = len(ids) - sum(min(args.parte, len(ids) - i) for i, _ in pendientes)
    with ProcessPoolExecutor(max_workers=args.procesos) as ejecutor:
        futuros = [ejecutor.submit(procesar_parte, i, x[i:i + args.parte], ids[i:i + args.parte], r)
                   for i, r in pendientes]
        for futuro in as_completed(futuros):
            _, n = futuro.result()
            hechos += n
            transcurrido = time.time() - t0
            print(f"{hechos:>7,} / {len(ids):,}  ({transcurrido / 60:.1f} min)", flush=True)

    unir(Path(args.salida) if args.salida else SALIDA, carpeta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limite", type=int, default=0, help="procesar solo n fragmentos al azar (prueba)")
    p.add_argument("--parte", type=int, default=500, help="fragmentos por parte")
    p.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--salida", help="ruta del archivo final (por omisión, datos/atributos_extendidos_subconjunto.csv.gz)")
    p.add_argument("--unir", action="store_true", help="solo juntar las partes ya calculadas")
    a = p.parse_args()
    if a.unir:
        unir(Path(a.salida) if a.salida else SALIDA)
    else:
        main(a)
