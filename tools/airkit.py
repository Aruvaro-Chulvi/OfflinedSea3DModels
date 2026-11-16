"""
airkit — propeller aircraft by code (MIT): fuselage, wings, tail, radial engines with propellers, landing gear.

One description of the aeroplane (sizes, the fuselage's sections, where the wings and the engines are) gives the
whole aircraft as built, or its parts one by one for a wreck script to break, crumple and scatter. Metres, +Y up,
nose to +Z, the ground under the wheels at y = 0 when it stands on its gear; the nose at z = 0 (wing_z, tail_z… are negative).

    from airkit import Plane, Engine
    p = Plane(length=18.9, span=29.25, …, engines=[Engine(x=0, z=…), …])
    parts = p.build()        # {"fuselage", "wing_l", "wing_r", "tail", "engine_0", …, "gear_l", "gear_r", "glass"}

Corrugated skin (the Junkers ribs) is in the geometry: the panels are ridged along the airflow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from meshkit import Mesh, merge, box, cyl, lathe, tube, sstep, fbm


@dataclass
class Engine:
    x: float                  # across (0 = the nose engine)
    y: float                  # height of the thrust line
    z: float                  # the propeller's plane
    r: float = 0.7            # cowling radius
    blades: int = 3
    prop: float = 1.6         # propeller radius
    nacelle: float = 3.0      # length of the nacelle behind the cowling


@dataclass
class Plane:
    length: float
    span: float
    fus_w: float              # fuselage width
    fus_h: float              # fuselage height
    fus_y: float              # height of the fuselage's centre line
    wing_z: float             # where the wing root's leading edge is (z, negative: behind the nose)
    root_chord: float
    tip_chord: float
    wing_y: float             # height of the wing root (low wing: near the fuselage bottom)
    dihedral: float = 0.05    # rad
    sweep: float = 0.0        # the leading edge's sweep back (m at the tip)
    thick: float = 0.16       # thickness / chord at the root
    tail_z: float = None      # leading edge of the tailplane
    tail_span: float = 8.0
    tail_chord: float = 2.0
    fin_h: float = 3.0
    fin_chord: float = 2.6
    engines: list = field(default_factory=list)
    gear: tuple = None        # (x of each main leg, z, wheel radius, leg length), the tail wheel under the tail
    corrugated: bool = False
    skin: tuple = (0.72, 0.72, 0.7)
    dark: tuple = (0.12, 0.12, 0.13)
    glass: tuple = (0.16, 0.22, 0.26)
    windows: tuple = ()       # (z0, z1, n, y, h) cabin windows on both sides
    cockpit: tuple = None     # (z0, z1) the windscreen
    nose: float = 0.25        # 0 blunt … 1 pointed
    upsweep: float = 0.4      # how much the tail cone rises (fraction of the fuselage height)

    # ── fuselage ─────────────────────────────────────────────────────────
    def zf(self, u):          # u 0 tail … 1 nose -> z (the nose at z = 0)
        return (u - 1.0) * self.length

    def section(self, u):
        """(half-width, half-height, centre y) of the fuselage at u (0 tail, 1 nose)."""
        k_nose = sstep(1.0, 0.86 - 0.08 * self.nose, u) ** (0.5 + 0.6 * self.nose)
        k_tail = 0.12 + 0.88 * sstep(0.0, 0.55, u) ** 0.9
        k = float(min(k_nose if u > 0.5 else 1.0, k_tail)) if u < 0.999 else 0.02
        cy = self.fus_y + self.fus_h * self.upsweep * 0.5 * float(sstep(0.55, 0.0, u))
        return self.fus_w / 2 * max(k, 0.05), self.fus_h / 2 * max(k, 0.06) * (1 - 0.35 * float(sstep(0.5, 0.0, u))), cy

    def fuselage(self):
        nu, na = 70, 28
        V, C = [], []
        us = np.linspace(0, 1, nu)
        for u in us:
            hw, hh, cy = self.section(u)
            z = self.zf(u)
            for j in range(na):
                a = 2 * math.pi * j / na
                ca, sa = math.cos(a), math.sin(a)
                sq = 3.2                                            # a boxy section (Junkers) … round
                r = 1.0 / (abs(ca) ** sq + abs(sa) ** sq) ** (1 / sq) if self.corrugated else 1.0
                x, y = hw * ca * r, hh * sa * r
                if self.corrugated:                                 # ribs along the airflow
                    rib = 0.022 * (0.5 + 0.5 * math.cos(j * math.pi))
                    x += rib * ca; y += rib * sa
                V.append((x, cy + y, z)); C.append(self.skin)
        f = []
        for i in range(nu - 1):
            for j in range(na):
                a = i * na + j; b = i * na + (j + 1) % na
                f += [(a, a + na, b), (b, a + na, b + na)]
        o = len(V); V.append((0, self.section(0)[2], self.zf(0) - 0.05)); C.append(self.skin)
        f += [(o, (j + 1) % na, j) for j in range(na)]
        o2 = len(V); V.append((0, self.section(1)[2], self.zf(1) + 0.02)); C.append(self.skin)
        last = (nu - 1) * na
        f += [(o2, last + j, last + (j + 1) % na) for j in range(na)]
        return Mesh(V, f, np.array(C))

    def windows_mesh(self):
        P = []
        if self.windows:
            z0, z1, n, y, h = self.windows
            for k in range(n):
                z = z0 + (z1 - z0) * k / max(1, n - 1)
                u = (z - self.zf(0)) / self.length
                hw, hh, cy = self.section(u)
                for s in (-1, 1):
                    P.append(box(0.05, h, 0.42, self.glass, bottom=False).T(s * (hw + 0.005), y, z))
        if self.cockpit:
            z0, z1 = self.cockpit
            u = (z1 - self.zf(0)) / self.length
            hw, hh, cy = self.section(u)
            ws = box(hw * 1.7, hh * 0.55, (z1 - z0), self.glass, bottom=False).T(0, cy + hh * 0.72, (z0 + z1) / 2)
            ws.v[:, 1] -= np.clip(ws.v[:, 2] - z0, 0, None) / max(z1 - z0, 0.1) * hh * 0.25
            P.append(ws)
        return merge(P) if P else Mesh()

    # ── wings and tail ─────────────────────────────────────────────────────
    def _airfoil(self, chord, thick, n=14):
        """Upper and lower surface points (z back from the leading edge, y) of a simple cambered section."""
        t = np.linspace(0, 1, n) ** 1.6
        yt = 5 * thick * (0.2969 * np.sqrt(t) - 0.126 * t - 0.3516 * t ** 2 + 0.2843 * t ** 3 - 0.1036 * t ** 4)
        camber = 0.03 * np.sin(np.pi * t)
        up = np.stack([-t * chord, (camber + yt) * chord], 1)
        lo = np.stack([-t * chord, (camber - yt) * chord], 1)
        return np.vstack([up, lo[::-1][1:-1]])

    def surface(self, x0, x1, z0, y0, c0, c1, th0, th1, sweep, dih, col, ribs=False, vertical=False, ns=16):
        """A lifting surface from x0 to x1 (root to tip): chord c, thickness, the leading edge swept back by sweep."""
        V, C = [], []
        prof0 = self._airfoil(1.0, th0)
        m = len(prof0)
        for i in range(ns + 1):
            s = i / ns
            x = x0 + (x1 - x0) * s
            c = c0 + (c1 - c0) * s
            th = th0 + (th1 - th0) * s
            prof = self._airfoil(c, th)
            zle = z0 - sweep * s
            yb = y0 + abs(x - x0) * math.tan(dih)
            for k, (pz, py) in enumerate(prof):
                if ribs:
                    py += 0.012 * math.cos(x / 0.09 * math.pi) * np.sign(py)   # corrugations across the span
                if vertical:
                    V.append((py, yb + (x - x0), zle + pz))
                else:
                    V.append((x, yb + py, zle + pz))
                C.append(col)
        f = []
        for i in range(ns):
            for k in range(m):
                a = i * m + k; b = i * m + (k + 1) % m
                f += [(a, b, a + m), (b, b + m, a + m)]
        # tip cap
        o = ns * m
        cen = len(V); V.append(tuple(np.mean(np.array(V[o:o + m]), 0))); C.append(col)
        f += [(cen, o + k, o + (k + 1) % m) for k in range(m)]
        return Mesh(V, f, np.array(C))

    def wings(self):
        out = {}
        hwr = self.fus_w / 2 * 0.9
        for s, name in ((-1, "wing_r"), (1, "wing_l")):
            out[name] = self.surface(s * hwr, s * self.span / 2, self.wing_z, self.wing_y,
                                     self.root_chord, self.tip_chord, self.thick, self.thick * 0.6, self.sweep,
                                     self.dihedral, self.skin, ribs=self.corrugated)
        return out

    def tail(self):
        tz = self.tail_z if self.tail_z is not None else self.zf(0) + self.tail_chord + 0.6
        cy = self.section(0.04)[2]
        P = []
        for s in (-1, 1):
            P.append(self.surface(0.0, s * self.tail_span / 2, tz, cy, self.tail_chord, self.tail_chord * 0.6, 0.1, 0.08,
                                  self.tail_chord * 0.15, 0.0, self.skin, ribs=self.corrugated, ns=8))
        P.append(self.surface(0.0, self.fin_h, tz + 0.4, cy + self.section(0.04)[1] * 0.5, self.fin_chord, self.fin_chord * 0.5,
                              0.1, 0.08, self.fin_chord * 0.35, 0.0, self.skin, ribs=self.corrugated, vertical=True, ns=8))
        return merge(P)

    # ── engines, gear ─────────────────────────────────────────────────────
    def engine(self, e, prop=True, spin=0.0):
        r = e.r
        cowl = lathe([(r * 0.55, 0.0), (r * 0.98, 0.15), (r, 0.45), (r * 0.96, 0.9), (r * 0.8, e.nacelle * 0.5),
                      (r * 0.35, e.nacelle)], self.skin, seg=20)
        cowl.R(-math.pi / 2, 0, 0)                                  # along −z from the front
        face = cyl(r * 0.55, r * 0.55, 0.05, self.dark, seg=20).R(math.pi / 2, 0, 0).T(0, 0, 0.02)
        P = [cowl, face]
        for k in range(9):                                          # the radial's cylinders seen through the cowling
            a = 2 * math.pi * k / 9
            P.append(cyl(0.09, 0.07, r * 0.45, self.dark, seg=6).R(0, 0, a - math.pi / 2).T(0, 0, -0.08))
        if prop:
            hub = lathe([(0.001, 0.45), (0.12, 0.38), (0.18, 0.15), (0.18, 0.0)], self.dark, seg=10).R(math.pi / 2, 0, 0).T(0, 0, 0.1)
            P.append(hub)
            for b in range(e.blades):
                a = spin + 2 * math.pi * b / e.blades
                bl = box(0.2, e.prop, 0.04, self.dark).R(0, 0.35, 0).R(0, 0, a).T(0, 0, 0.18)
                P.append(bl)
        m = merge(P)
        return m.T(e.x, e.y, e.z)

    def gear_leg(self, x, z, wr, ll, spat=True):
        y_ax = wr
        P = [cyl(wr, wr, 0.28, self.dark, seg=16).R(0, 0, math.pi / 2).T(x + 0.14, y_ax, z)]
        P.append(tube([(x, y_ax, z), (x * 0.7, y_ax + ll, z + 0.2)], 0.08, self.skin, seg=6))
        P.append(tube([(x, y_ax, z), (x * 1.0 - math.copysign(0.6, x), y_ax + ll * 0.9, z - 0.6)], 0.06, self.skin, seg=6))
        if spat:
            sp = lathe([(0.001, -wr * 1.3), (0.2, -wr * 1.1), (0.24, 0.0), (0.2, wr * 1.0), (0.001, wr * 1.4)], self.skin, seg=10)
            sp.R(math.pi / 2, 0, 0).S(1.0, 1.25, 1.0).T(x, y_ax + 0.05, z)
            P.append(sp)
        return merge(P)

    def build(self):
        out = {"fuselage": self.fuselage(), "glass": self.windows_mesh(), "tail": self.tail()}
        out.update(self.wings())
        for k, e in enumerate(self.engines):
            out[f"engine_{k}"] = self.engine(e, spin=0.3 * k)
        if self.gear:
            xg, zg, wr, ll, spat = self.gear
            out["gear_l"] = self.gear_leg(xg, zg, wr, ll, spat)
            out["gear_r"] = self.gear_leg(-xg, zg, wr, ll, spat)
            tz = self.zf(0.06)
            yb = self.section(0.06)[2] - self.section(0.06)[1] * 0.8      # under the tail; build() leaves it in the air:
            ya = yb - 0.55                                               # stand the aircraft with standing()
            out["tail_wheel"] = merge([cyl(wr * 0.4, wr * 0.4, 0.15, self.dark, seg=10).R(0, 0, math.pi / 2).T(0.07, ya, tz),
                                       tube([(0, ya, tz), (0, yb + 0.1, tz + 0.3)], 0.05, self.skin, seg=5)])
        return out


def decal(text, h, col, depth=0.01):
    """Letters for registrations and names (sailkit's 5×7 letters), reading towards +x on a face that looks to −z…
    callers rotate them onto the skin."""
    from sailkit import lettering
    return lettering(text, h, col, depth=depth)


def standing(parts, z_main):
    """The aircraft on the ground: rotated about the main wheels (at z_main) until the tail wheel touches, then set
    down on y = 0. parts: the dict from build() (and anything added to it)."""
    tw = parts["tail_wheel"].v
    i = np.argmin(tw[:, 1])
    lo_main = min(m.v[:, 1].min() for k, m in parts.items() if k.startswith("gear") and len(m.v))
    a = math.atan2(tw[:, 1].min() - lo_main, z_main - tw[i, 2])
    for m in parts.values():
        if len(m.v):
            m.T(0, 0, -z_main).R(-a, 0, 0).T(0, 0, z_main)
    lo = min(m.v[:, 1].min() for m in parts.values() if len(m.v))
    for m in parts.values():
        if len(m.v):
            m.T(0, -lo, 0)
    return parts
