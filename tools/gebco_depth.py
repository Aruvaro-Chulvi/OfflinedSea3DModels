"""
Depth (or height) of any point on Earth from a GEBCO Terrarium PMTiles file.

The file is the same one Offlined uses for its seafloor maps ("seafloor.pmtiles"):
PMTiles v3, WebP tiles encoded as Mapzen Terrarium (h = R*256 + G + B/256 - 32768).
Only the standard library and Pillow are needed.

    from gebco_depth import Gebco
    g = Gebco("path/to/seafloor.pmtiles")
    g.depth(41.7325, -49.9469)   # -> about -3800 (metres, negative = below sea level)
"""
from __future__ import annotations

import gzip
import io
import math
import struct
from functools import lru_cache

from PIL import Image


def _varint(b: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        c = b[i]; i += 1
        n |= (c & 0x7F) << shift
        if c < 0x80:
            return n, i
        shift += 7


def _zxy_to_id(z: int, x: int, y: int) -> int:
    """Tile id of PMTiles v3 (Hilbert curve inside each zoom level)."""
    acc = sum(4 ** k for k in range(z))
    n = 1 << z
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        s >>= 1
    return acc + d


class Gebco:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        h = self.f.read(127)
        if h[:7] != b"PMTiles" or h[7] != 3:
            raise ValueError("not a PMTiles v3 file")
        (root_off, root_len, _mo, _ml, leaf_off, _ll, data_off, _dl, _a, _e, _c, _cl,
         self.icomp, _tc, _tt, _minz, self.maxz) = struct.unpack("<QQQQQQQQQQQBBBBBB", h[8:102])
        self.data_off, self.leaf_off = data_off, leaf_off
        self.root = self._dir(root_off, root_len)

    def _read(self, off: int, n: int) -> bytes:
        self.f.seek(off)
        return self.f.read(n)

    def _dir(self, off: int, n: int) -> list[tuple[int, int, int, int]]:
        b = self._read(off, n)
        if self.icomp == 2:
            b = gzip.decompress(b)
        cnt, i = _varint(b, 0)
        ids, runs, lens, offs = [], [], [], []
        last = 0
        for _ in range(cnt):
            v, i = _varint(b, i); last += v; ids.append(last)
        for _ in range(cnt):
            v, i = _varint(b, i); runs.append(v)
        for _ in range(cnt):
            v, i = _varint(b, i); lens.append(v)
        for k in range(cnt):
            v, i = _varint(b, i)
            offs.append(offs[k - 1] + lens[k - 1] if v == 0 and k > 0 else v - 1)
        return list(zip(ids, runs, offs, lens))

    def _find(self, tid: int, d) -> tuple[int, int] | None:
        lo, hi = 0, len(d) - 1                         # last entry whose id is <= tid
        while lo <= hi:
            m = (lo + hi) // 2
            if d[m][0] <= tid:
                lo = m + 1
            else:
                hi = m - 1
        k = hi
        if k < 0:
            return None
        eid, run, off, ln = d[k]
        if run == 0:                                   # leaf directory
            return self._find(tid, self._dir(self.leaf_off + off, ln))
        if tid - eid < run:
            return off, ln
        return None

    @lru_cache(maxsize=256)
    def _tile(self, z: int, x: int, y: int):
        r = self._find(_zxy_to_id(z, x, y), self.root)
        if not r:
            return None
        im = Image.open(io.BytesIO(self._read(self.data_off + r[0], r[1]))).convert("RGB")
        return im.size[0], im.load()

    def depth(self, lat: float, lon: float) -> float | None:
        """Metres above (+) or below (-) sea level, bilinear at the finest zoom level."""
        z = self.maxz
        n = 1 << z
        lat = max(-85.05, min(85.05, lat))
        fx = (lon + 180.0) / 360.0 * n
        fy = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
        t = self._tile(z, int(fx) % n, int(fy))
        if not t:
            return None
        size, px = t
        u = (fx - int(fx)) * size - 0.5
        v = (fy - int(fy)) * size - 0.5
        i0, j0 = max(0, min(size - 2, int(math.floor(u)))), max(0, min(size - 2, int(math.floor(v))))
        a, b = min(1.0, max(0.0, u - i0)), min(1.0, max(0.0, v - j0))

        def h(i, j):
            r, g, bb = px[i, j]
            return r * 256 + g + bb / 256 - 32768

        return (h(i0, j0) * (1 - a) * (1 - b) + h(i0 + 1, j0) * a * (1 - b)
                + h(i0, j0 + 1) * (1 - a) * b + h(i0 + 1, j0 + 1) * a * b)


if __name__ == "__main__":
    import sys
    g = Gebco(sys.argv[1])
    for lat, lon, name in [(41.7325, -49.9469, "Titanic"), (27.8136, 33.9206, "Thistlegorm"),
                           (11.35, 142.2, "Challenger Deep"), (28.27, -16.64, "Teide")]:
        print(f"{name}: {g.depth(lat, lon):.0f} m")
