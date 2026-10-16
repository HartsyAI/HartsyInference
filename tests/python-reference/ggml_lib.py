"""ctypes binding to llama.cpp's libggml-base: its C quantizer and dequantizers, the ground truth for GGUF block formats."""
import ctypes

import numpy as np


class _TypeTraits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
_F32P = ctypes.POINTER(ctypes.c_float)


class GgmlLib:
    """Loads libggml-base (built with llama.cpp) and exposes ggml_quantize_chunk and type_traits.to_float on numpy arrays."""

    def __init__(self, path: str):
        self._lib = ctypes.CDLL(path)
        self._lib.ggml_get_type_traits.restype = ctypes.POINTER(_TypeTraits)
        self._lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, _F32P, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, _F32P]
        self._lib.ggml_quantize_init.argtypes = [ctypes.c_int]

    def traits(self, type_id: int) -> _TypeTraits:
        return self._lib.ggml_get_type_traits(type_id).contents

    def quantize(self, type_id: int, rows: np.ndarray, imatrix: np.ndarray) -> np.ndarray:
        """ggml_quantize_chunk over float32 rows [n, k]; returns the blocks as bytes."""
        t = self.traits(type_id)
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        out = np.zeros(rows.size // t.blck_size * t.type_size, dtype=np.uint8)
        self._lib.ggml_quantize_init(type_id)
        written = self._lib.ggml_quantize_chunk(type_id, rows.ctypes.data_as(_F32P), out.ctypes.data, 0, rows.shape[0], rows.shape[1],
                                                imatrix.ctypes.data_as(_F32P))
        if written != out.size:
            raise RuntimeError(f"ggml_quantize_chunk wrote {written} bytes, expected {out.size}")
        return out

    def dequantize(self, type_id: int, blocks: np.ndarray, count: int) -> np.ndarray:
        """type_traits.to_float over count values of raw blocks."""
        t = self.traits(type_id)
        if not t.to_float:
            raise ValueError(f"ggml type {type_id} has no dequantizer")
        blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
        out = np.zeros(count, dtype=np.float32)
        _TO_FLOAT(t.to_float)(blocks.ctypes.data, out.ctypes.data_as(_F32P), count)
        return out
