"""Point geometry encodings built with numpy (no shapely)."""

import numpy as np
import pyarrow as pa

_WKB_POINT = np.dtype([("order", "u1"), ("type", "<u4"), ("x", "<f8"), ("y", "<f8")])


def points_to_wkb(lon: np.ndarray, lat: np.ndarray) -> pa.BinaryArray:
    """Little-endian WKB Points, 21 bytes each."""
    n = len(lon)
    rec = np.empty(n, dtype=_WKB_POINT)
    rec["order"], rec["type"], rec["x"], rec["y"] = 1, 1, lon, lat
    offsets = np.arange(n + 1, dtype=np.int32) * _WKB_POINT.itemsize
    return pa.BinaryArray.from_buffers(
        pa.binary(), n, [None, pa.py_buffer(offsets), pa.py_buffer(rec.tobytes())]
    )


def bbox_struct(lon: np.ndarray, lat: np.ndarray) -> pa.StructArray:
    """GeoParquet covering struct; a point's box is the point itself."""
    x, y = pa.array(lon, pa.float64()), pa.array(lat, pa.float64())
    return pa.StructArray.from_arrays([x, y, x, y], ["xmin", "ymin", "xmax", "ymax"])
