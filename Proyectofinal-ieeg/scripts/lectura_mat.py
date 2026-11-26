"""Lectura de los fragmentos .mat del dataset Mayo–FNUSA directamente desde el zip.

Los archivos son MAT de versión 5 con un único arreglo `data` de 1 × 15000 (float32),
comprimido con zlib. Este lector solo usa la biblioteca estándar y NumPy, así que no
hace falta descomprimir los zips (8.5 GB) ni instalar SciPy para leerlos.
"""

import struct
import zipfile
import zlib

import numpy as np

# Códigos de tipo de dato de MAT v5 -> tipo de NumPy
_TIPOS_MAT = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
              7: "f4", 9: "f8", 12: "i8", 13: "u8"}
_MI_MATRIX = 14
_MI_COMPRESSED = 15


def _elementos(buf, inicio=0):
    """Recorre los elementos (etiqueta, datos) de un bloque MAT v5."""
    salida = []
    pos = inicio
    while pos < len(buf):
        tipo, n_bytes = struct.unpack_from("<II", buf, pos)
        if tipo >> 16:  # elemento pequeño: tamaño y tipo comparten 4 bytes
            n_bytes = tipo >> 16
            tipo &= 0xFFFF
            salida.append((tipo, buf[pos + 4:pos + 4 + n_bytes]))
            pos += 8
        else:
            salida.append((tipo, buf[pos + 8:pos + 8 + n_bytes]))
            pos += 8 + n_bytes + ((8 - n_bytes % 8) % 8)  # relleno a 8 bytes
    return salida


def leer_mat(contenido):
    """Devuelve el primer arreglo numérico de un archivo MAT v5 como vector float32."""
    for tipo, datos in _elementos(contenido, 128):  # 128 bytes de encabezado
        if tipo == _MI_COMPRESSED:
            tipo, datos = _elementos(zlib.decompress(datos))[0]
        if tipo == _MI_MATRIX:
            sub = _elementos(datos)
            # sub[0] = banderas, sub[1] = dimensiones, sub[2] = nombre, sub[3] = datos reales
            tipo_dato, valores = sub[3]
            return np.frombuffer(valores, _TIPOS_MAT[tipo_dato]).astype(np.float32)
    raise ValueError("No se encontró un arreglo en el archivo MAT")


class LectorZip:
    """Acceso por identificador de fragmento a los .mat y al segments.csv de un zip."""

    def __init__(self, ruta_zip):
        self.zip = zipfile.ZipFile(ruta_zip)
        nombres = self.zip.namelist()
        self.carpeta = nombres[0].split("/")[0]  # DATASET_FNUSA o DATASET_MAYO
        self.ids = sorted(n.split("/")[-1][:-4] for n in nombres if n.endswith(".mat"))

    def leer(self, segment_id):
        """Señal de un fragmento (15000 muestras)."""
        return leer_mat(self.zip.read(f"{self.carpeta}/{segment_id}.mat"))

    def segments_csv(self):
        """Contenido en bytes del archivo de metadatos segments.csv."""
        return self.zip.read(f"{self.carpeta}/segments.csv")
