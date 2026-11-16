"""
sailkit — wooden sailing ships by code (MIT): hull, castles, decks, masts, yards, rigging and sails.

One description of the ship (its measures, the shape of its hull, its castles and rig) gives the whole ship as
she sailed, or — built=False — what is left of her standing (no sails, no rigging, the masts as stumps), for the
wreck scripts to break and scatter. Metres, +Y up, bow to +Z, keel on y = 0, centred on her length.

    from sailkit import Ship, Mast
    s = Ship(L=48, B=13.7, D=6.4, draft=4.6, fore=(10, 3.0), aft=(14, 5.0), masts=[Mast(0.62, 34, "square", 3), …])
    p = s.build(built=True)      # {"hull", "deck", "spars", "ropes", "sails", "guns"}: Mesh

Helpers for the wrecks: solidify (a thickness for a broken shell), frames (the ribs left where the planking is
gone), cannon, gun_carriage, cannonball.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, sstep, fbm


@dataclass
class Mast:
    at: float                 # position along the ship, 0 = stern, 1 = bow
    height: float             # above the deck it stands on
    rig: str = "square"       # "square" (yards), "gaff" (fore-and-aft), "lateen"
    yards: int = 2            # square sails one above the other
    top: bool = True          # a top (platform) where the lower mast ends


@dataclass
class Ship:
    L: float                  # length on deck
    B: float                  # greatest beam
    D: float                  # keel to the main deck amidships
    draft: float
    fore: tuple = (0.0, 0.0)           # forecastle: (length, height above the main deck), or a list of tiers
    aft: tuple = (0.0, 0.0)            # aftercastle: the same; [(16, 2.6), (9, 2.4)] = a quarterdeck and a poop above it
    sheer: float = 1.5                 # how much the deck rises towards the ends
    tumble: float = 0.12               # tumblehome: the sides lean in above the waterline (fraction of the beam)
    castle_tumble: float = 0.12        # and the castles' sides, more
    bluff: float = 0.6                 # 0 fine ends … 1 round, full bow and stern
    transom: float = 0.45              # width of the flat stern, fraction of the beam
    bowsprit: float = 0.0              # length
    spritsail: bool = True             # a square sail under the bowsprit (to the 17th century)
    jibs: int = 0                      # triangular headsails from the bowsprit to the foremast (later ships)
    yard: float = 1.6                  # the lowest yard's length, in beams
    masts: list = field(default_factory=list)
    gunports: list = field(default_factory=list)    # [(height above the keel amidships, from t, to t, count)]
    guns: bool = True                               # muzzles showing at the ports (as built)
    stern_windows: int = 0
    hull_col: tuple = (0.3, 0.22, 0.15)             # below the wales (tarred)
    upper_col: tuple = (0.48, 0.35, 0.22)
    wale_col: tuple = (0.12, 0.1, 0.08)
    deck_col: tuple = (0.58, 0.47, 0.33)
    trim_col: tuple = (0.66, 0.5, 0.18)             # carving, gilding
    port_col: tuple = (0.55, 0.13, 0.08)            # the gunport lids
    sail_col: tuple = (0.86, 0.82, 0.7)
    spar_col: tuple = (0.4, 0.29, 0.18)
    rope_col: tuple = (0.2, 0.16, 0.12)
    lower_decks: list = field(default_factory=list) # heights above the keel amidships of the decks below the main one
    deckhouses: list = field(default_factory=list)  # [(t0, t1, width, height, colour)]
    funnel: tuple = None                            # (t, radius, height, colour)
    rails: bool = True

    # ── the lines ──────────────────────────────────────────────────────────
    def zt(self, t):                    # t (0 stern … 1 bow) -> z
        return (t - 0.5) * self.L

    def tz(self, z):
        return z / self.L + 0.5

    def w(self, t):                     # half-beam factor along the ship
        p = 1.7 - self.bluff
        bow = math.sin(min(1.0, (1 - t) / 0.3) * math.pi / 2) ** p
        stern = self.transom + (1 - self.transom) * math.sin(min(1.0, t / 0.32) * math.pi / 2) ** p
        return max(0.004, min(bow, stern))

    def keel(self, t):                  # the keel line: it curves up into the stem at the bow, a little at the stern
        return self.D * 0.62 * sstep(0.8, 1.0, t) ** 1.5 + self.D * 0.18 * sstep(0.1, 0.0, t) ** 1.3

    def deck(self, t):                  # the main deck, with its sheer
        return self.D + self.sheer * ((2 * t - 1) ** 2)

    def tiers(self):                    # -> [(t of the front, height of this tier, its base above the deck, +1 bow / -1 stern)]
        out = []
        for spec, sgn in ((self.fore, 1), (self.aft, -1)):
            tl = [spec] if spec and not isinstance(spec[0], (tuple, list)) else list(spec)
            base = 0.0
            for ln, hh in tl:
                if ln > 0:
                    out.append((1 - ln / self.L if sgn > 0 else ln / self.L, hh, base, sgn)); base += hh
        return out

    def castle(self, t):                # height of the castles above the main deck at t
        h = 0.0
        for a, hh, base, sgn in self.tiers():
            h += hh * float(sstep(a - 0.004 * sgn, a + 0.004 * sgn, t))
        return h

    def half(self, t, y):               # half-width at height y: a round bilge, widest at the wales, tumblehome above
        yb, yd = self.keel(t), self.deck(t)
        y = np.asarray(y, float)
        u = np.clip((y - yb) / max(0.3, yd - yb), 0, None)
        bilge = 1 - (1 - np.minimum(u / 0.62, 1)) ** 2.6
        tum = 1 - self.tumble * sstep(0.62, 1.0, u)
        over = np.clip((y - yd) / max(1.0, self.castle(t) + 1.0), 0, 1)     # above the deck: the castles lean in
        tum = tum * (1 - self.castle_tumble * over)
        return self.B / 2 * self.w(t) * bilge * tum

    def top(self, t):                   # top of the side (bulwark or castle rail)
        return self.deck(t) + self.castle(t) + 1.0

    # ── building ───────────────────────────────────────────────────────────
    def _hull(self):
        """Both sides from the keel up to the rail, as one surface, with the flat transom closed."""
        ns, nv = 120, 26
        ts = np.concatenate([np.linspace(0.0, 0.94, ns - 20), np.linspace(0.945, 1.0, 21)])
        W = 2 * nv + 1
        V, C = [], []
        for t in ts:
            yb, yt = self.keel(t), self.top(t)
            u = np.abs(nv - np.arange(W)) / nv
            s = np.where(np.arange(W) < nv, -1.0, 1.0)
            uu = u ** 0.85
            y = yb + (yt - yb) * uu
            x = s * self.half(t, y)
            x[nv] = 0.0
            V.append(np.stack([x, y, np.full(W, self.zt(t))], 1))
            col = np.where((y < self.draft + 0.25)[:, None], np.array(self.hull_col)[None], np.array(self.upper_col)[None])
            C.append(col)
        V, C = np.vstack(V), np.vstack(C)
        f = []
        for i in range(len(ts) - 1):
            a = i * W + np.arange(W - 1); b = a + W
            f.append(np.stack([a, a + 1, b], 1)); f.append(np.stack([b, a + 1, b + 1], 1))
        f = np.vstack(f)
        # the transom: a fan from its middle, closed across the top
        o = len(V)
        V = np.vstack([V, [[0.0, (self.keel(0) + self.top(0)) / 2, self.zt(0)]]]); C = np.vstack([C, [self.upper_col]])
        fan = [(o, j + 1, j) for j in range(W - 1)] + [(o, 0, W - 1)]
        m = Mesh(V, np.vstack([f, np.array(fan)]), C)
        m.c = np.clip(m.c * (0.9 + 0.2 * fbm(m.v, 0.25, 3, 3))[:, None], 0, 1)
        return m

    def _decks(self):
        """The main deck and the castle decks above it (each a planked surface), the castles' fronts with doors."""
        P = []
        for kind in ["main", "castle"] + list(self.lower_decks):
            v, f, prev = [], [], None
            for t in np.linspace(0.015, 0.975, 90):
                h = 0.0 if kind in ("main", "castle") else 0.0
                if kind == "castle":
                    h = self.castle(t)
                    if h < 0.5:
                        prev = None; continue
                y = self.deck(t) + h if isinstance(kind, str) else kind + (self.deck(t) - self.D) * 0.85
                if not isinstance(kind, str) and y < self.keel(t) + 1.0:
                    prev = None; continue
                x = float(self.half(t, y)) * 0.985
                cur = len(v); v += [(-x, y, self.zt(t)), (x, y, self.zt(t))]
                if prev is not None:
                    f += [(prev, cur, prev + 1), (prev + 1, cur, cur + 1)]
                prev = cur
            if f:
                P.append(Mesh(v, f, self.deck_col))
        for t, hh, base, side in self.tiers():
            z = self.zt(t); y = self.deck(t) + base
            x = float(self.half(t, y + hh)) * 0.98
            P.append(box(2 * x, hh + 1.0, 0.3, self.upper_col).T(0, y, z))
            sgn = -side                                                    # the side facing the waist
            for k, dx in enumerate((-x * 0.45, x * 0.45)):
                P.append(quad(dx, y + 0.95, z + sgn * 0.15, 0.9, 1.8, "+z" if sgn > 0 else "-z", (0.1, 0.07, 0.05)))
            P.append(box(2 * x, 0.18, 0.5, self.trim_col).T(0, y + hh - 0.1, z + sgn * 0.1))
        return merge(P)

    def _rails(self):
        P = []
        for s in (-1, 1):
            ts = np.linspace(0.02, 0.985, 80)
            pts = [(s * float(self.half(t, self.top(t))) * 0.99, self.top(t), self.zt(t)) for t in ts]
            P.append(tube(pts, 0.12, self.spar_col, seg=4))
        return merge(P)

    def _details(self, built):
        """Wales, gunports and guns, stern windows and carving, rudder, deckhouses, funnel."""
        P, G = [], []
        for k, dy in enumerate((0.3, 1.5)):                                  # two wales along each side
            for s in (-1, 1):
                pts = []
                for t in np.linspace(0.01, 0.995, 70):
                    y = self.draft + dy + (self.deck(t) - self.D) * 0.85
                    pts.append((s * (float(self.half(t, y)) + 0.07), y, self.zt(t)))
                P.append(tube(pts, 0.17, self.wale_col, seg=5))
        for hy, t0, t1, n in self.gunports:
            for t in np.linspace(t0, t1, n):
                y = hy + (self.deck(t) - self.D) * 0.85
                for s in (-1, 1):
                    x = s * float(self.half(t, y))
                    nm = "+x" if s > 0 else "-x"
                    P.append(quad(x, y, self.zt(t), 0.95, 0.9, nm, self.port_col, eps=0.05))
                    P.append(quad(x, y, self.zt(t), 0.7, 0.66, nm, (0.05, 0.04, 0.03), eps=0.07))
                    if built and self.guns:
                        g = cannon(1.4, 0.07, True).R(0, s * math.pi / 2, 0).T(x - s * 0.5, y - 0.05, self.zt(t))
                        G.append(g)
        if self.stern_windows:
            z = self.zt(0) - 0.06
            y0 = self.deck(0.01) + self.castle(0.01) * 0.55
            hw = float(self.half(0.0, y0))
            for k in range(self.stern_windows):
                x = (k - (self.stern_windows - 1) / 2) * min(1.3, 1.7 * hw / self.stern_windows)
                P.append(quad(x, y0, z, 0.7, 1.0, "-z", (0.12, 0.14, 0.15), eps=0.02))
                P.append(quad(x, y0, z, 0.9, 1.25, "-z", self.trim_col, eps=0.01))
            P.append(box(2 * hw * 0.95, 0.35, 0.25, self.trim_col).T(0, y0 + 0.9, z))
            P.append(box(2 * hw * 0.95, 0.35, 0.25, self.trim_col).T(0, y0 - 0.95, z))
        # the rudder, hung on the sternpost
        yb, yt = 0.15, self.deck(0.0) + 0.5
        P.append(merge([box(0.38, yt - yb, 1.4, self.hull_col).T(0, yb, self.zt(0) - 0.7),
                        box(0.3, 0.3, 3.0, self.spar_col).T(0, yt - 0.2, self.zt(0) + 0.6)]))
        for t0, t1, wd, ht, col in self.deckhouses:
            tm = (t0 + t1) / 2
            y = self.deck(tm) + self.castle(tm)
            P.append(box(wd, ht, (t1 - t0) * self.L, col).T(0, y, self.zt(tm)))
            P.append(box(wd + 0.3, 0.15, (t1 - t0) * self.L + 0.3, self.spar_col).T(0, y + ht, self.zt(tm)))
            n = max(1, int((t1 - t0) * self.L / 1.9))
            for k in range(n):
                zz = self.zt(t0) + (k + 0.5) * (t1 - t0) * self.L / n
                for s in (-1, 1):
                    P.append(quad(s * wd / 2, y + ht * 0.62, zz, 0.55, 0.55, "+x" if s > 0 else "-x", (0.1, 0.12, 0.13), eps=0.02))
        if self.funnel:
            t, r, h, col = self.funnel
            y = self.deck(t)
            P.append(cyl(r, r, h, col, seg=16).T(0, y, self.zt(t)))
            P.append(cyl(r * 1.04, r * 1.04, h * 0.18, (0.07, 0.07, 0.07), seg=16, caps=False).T(0, y + h * 0.82, self.zt(t)))
        return merge(P), (merge(G) if G else Mesh())

    def _mast_base(self, m):
        return self.deck(m.at) + self.castle(m.at)

    def _rig(self, built):
        """Masts, tops, yards, bowsprit; as built also the standing rigging and the sails."""
        spars, ropes, sails = [], [], []
        for m in self.masts:
            z = self.zt(m.at); y0 = self._mast_base(m)
            h = m.height
            r0 = 0.011 * h + 0.14
            if not built:                                                  # what stays standing: a stump
                spars.append(cyl(r0, r0 * 0.9, 2.2, self.spar_col, seg=10).T(0, y0, z))
                continue
            lower = 0.58 if m.top else 1.0
            spars.append(cyl(r0, r0 * 0.75, h * lower + 0.6, self.spar_col, seg=10).T(0, y0, z))
            if m.top:
                spars.append(cyl(r0 * 0.6, r0 * 0.3, h * (1 - lower) + 0.3, self.spar_col, seg=8).T(0, y0 + h * lower - 0.6, z - r0 * 1.3))
                spars.append(cyl(1.1 + 0.02 * h, 0.9 + 0.02 * h, 0.55, self.spar_col, seg=14).T(0, y0 + h * lower - 0.6, z))
            ht = lambda k: y0 + h * k
            half_b = float(self.half(m.at, self.top(m.at))) + 0.25
            yr = self.top(m.at) - 0.6
            for s in (-1, 1):                                              # shrouds, with ratlines
                feet = [(s * half_b, yr, z + dz) for dz in (-1.6, -0.6, 0.4)]
                head = (s * 0.5, ht(lower) - 0.8, z)
                for p in feet:
                    ropes.append(tube([p, head], 0.03, self.rope_col, seg=3))
                for k in range(1, 9):
                    a = k / 9.5
                    pts = [tuple(np.array(p) * (1 - a) + np.array(head) * a) for p in feet]
                    ropes.append(tube(pts, 0.015, self.rope_col, seg=3))
            if m.rig == "square":
                ys = [ht(lower * (0.45 if m.yards > 1 else 0.8))] + [ht(lower + (1 - lower) * (k / max(1, m.yards - 1)) * 0.82) for k in range(1, m.yards)]
                if m.yards > 1:
                    ys[1] = max(ys[1], ht(lower) + 0.8)
                spans = [self.B * self.yard * (1.0 - 0.2 * k) * (0.85 if m.at > 0.75 else 1.0) for k in range(m.yards)]
                for k, (yk, sp) in enumerate(zip(ys, spans)):
                    spars.append(cyl(0.14 + 0.004 * sp, 0.1, sp, self.spar_col, seg=6).R(0, 0, math.pi / 2).T(sp / 2, yk, z + 0.55))
                    y_bot = ys[k - 1] + 0.5 if k else max(self.top(m.at) + 0.8, yk - h * 0.32)
                    w_bot = spans[k - 1] * 0.97 if k else sp * 1.02
                    sails.append(self._square_sail(sp * 0.96, w_bot, yk - 0.15, y_bot, z + 0.7))
            elif m.rig == "gaff":
                bl = min(self.L * 0.22, h * 0.55)
                yb = self.top(m.at) + 1.2
                spars.append(cyl(0.13, 0.1, bl, self.spar_col, seg=6).R(-math.pi / 2, 0, 0).T(0, yb, z - 0.3))
                a = 0.5
                gl = bl * 0.75
                yg = ht(lower) - 1.0
                spars.append(cyl(0.12, 0.09, gl, self.spar_col, seg=6).R(-(math.pi / 2 - a), 0, 0).T(0, yg, z - 0.3))
                peak = (0.0, yg + gl * math.sin(a), z - 0.3 - gl * math.cos(a))
                sails.append(self._panel([(0, yb + 0.1, z - 0.4), (0, yg - 0.1, z - 0.4), peak, (0, yb + 0.1, z - 0.3 - bl)]))
            elif m.rig == "lateen":
                ln = h * 0.92; a = 0.72
                yc = ht(0.42)
                d = np.array([0.0, math.sin(a), -math.cos(a)])
                lo_end = np.array([0.0, yc, z + 0.45]) - d * ln * 0.38
                hi_end = np.array([0.0, yc, z + 0.45]) + d * ln * 0.62
                spars.append(tube([tuple(lo_end), tuple(hi_end)], 0.12, self.spar_col, seg=6))
                clew = (0.0, self.top(m.at) + 1.4, z - ln * 0.42)
                sails.append(self._panel([tuple(lo_end + [0, -0.1, 0]), tuple(hi_end + [0, -0.2, 0]), clew], belly=0.6))
        if self.bowsprit:
            tb = 0.975
            a = 0.42
            base = np.array([0.0, self.deck(tb) + self.castle(tb) * 0.4 + 0.3, self.zt(tb) - 2.0])
            dirv = np.array([0.0, math.sin(a), math.cos(a)])
            tip = base + dirv * self.bowsprit
            spars.append(tube([tuple(base), tuple(tip)], 0.26, self.spar_col, seg=8))
            if built and self.masts:
                fm = max(self.masts, key=lambda m: m.at)
                lower = 0.58 if fm.top else 1.0
                ropes.append(tube([tuple(tip), (0, self._mast_base(fm) + fm.height * lower - 0.6, self.zt(fm.at))], 0.06, self.rope_col, seg=3))
                ropes.append(tube([tuple(tip), (0, self._mast_base(fm) + fm.height * 0.97, self.zt(fm.at))], 0.04, self.rope_col, seg=3))
                fl = 0.58 if fm.top else 1.0
                for k in range(self.jibs):                                 # jibs: tack on the bowsprit, head up the foremast stay
                    a_ = (k + 1) / (self.jibs + 0.5)
                    tack = base + dirv * self.bowsprit * (0.35 + 0.6 * a_)
                    mh = np.array([0.0, self._mast_base(fm) + fm.height * (fl - 0.05 + 0.35 * a_), self.zt(fm.at)])
                    head = tack + (mh - tack) * 0.88
                    clew = np.array([0.0, self.top(0.93) + 0.8 + 0.6 * k, self.zt(fm.at) + 1.5 + (tack[2] - self.zt(fm.at)) * 0.25])
                    sails.append(self._panel([tuple(tack), tuple(head), tuple(clew)], belly=0.35, n=6))
                if fm.rig == "square" and self.spritsail:                  # the spritsail under the bowsprit
                    mid = base + dirv * self.bowsprit * 0.72
                    sp = self.B * 0.6
                    spars.append(cyl(0.12, 0.09, sp, self.spar_col, seg=6).R(0, 0, math.pi / 2).T(sp / 2, mid[1] - 0.3, mid[2]))
                    sails.append(self._square_sail(sp * 0.95, sp * 0.95, mid[1] - 0.4, mid[1] - 3.2, mid[2] + 0.2, belly=0.5))
        if built and len(self.masts) > 1:                                  # stays from each masthead down to the next mast forward
            ms = sorted(self.masts, key=lambda m: m.at)
            for a, b in zip(ms[:-1], ms[1:]):
                la = 0.58 if a.top else 1.0
                ropes.append(tube([(0, self._mast_base(a) + a.height * la - 0.7, self.zt(a.at)), (0, self._mast_base(b) + 1.5, self.zt(b.at))], 0.05, self.rope_col, seg=3))
                ropes.append(tube([(0, self._mast_base(a) + a.height * 0.96, self.zt(a.at)), (0, self._mast_base(b) + b.height * 0.58, self.zt(b.at))], 0.035, self.rope_col, seg=3))
        return merge(spars), (merge(ropes) if ropes else Mesh()), (merge(sails) if sails else Mesh())

    def _square_sail(self, w_top, w_bot, y_top, y_bot, z, belly=1.0):
        nx, ny = 10, 7
        v, f = [], []
        for j in range(ny + 1):
            u = j / ny
            wd = w_top + (w_bot - w_top) * u
            y = y_top + (y_bot - y_top) * u
            for i in range(nx + 1):
                s = i / nx - 0.5
                bul = belly * (1 - (2 * s) ** 2) * math.sin(math.pi * (0.15 + 0.85 * u))
                v.append((s * wd, y - 0.25 * (1 - (2 * s) ** 2) * u * u, z + bul))
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i; b = a + nx + 1
                f += [(a, b, a + 1), (a + 1, b, b + 1)]
        m = Mesh(v, f, self.sail_col)
        cl = np.array(v)
        seams = 0.94 + 0.06 * np.cos(cl[:, 0] * math.pi / 0.6) ** 8                  # the cloths sewn side by side
        m.c = np.clip(m.c * (seams * (0.9 + 0.12 * fbm(m.v, 0.25, 5, 2)))[:, None], 0, 1)
        return m

    def _panel(self, pts, belly=0.4, n=8):
        """A fore-and-aft sail (3 or 4 corners) with a little belly to port."""
        pts = [np.array(p, float) for p in pts]
        if len(pts) == 3:
            pts = [pts[0], pts[1], pts[2], pts[2]]
        a, b, c, d = pts
        v, f = [], []
        for j in range(n + 1):
            u = j / n
            l, r = a + (b - a) * u, d + (c - d) * u
            for i in range(n + 1):
                s = i / n
                p = l + (r - l) * s
                p = p + np.array([belly * math.sin(math.pi * s) * math.sin(math.pi * u), 0, 0])
                v.append(p)
        for j in range(n):
            for i in range(n):
                q = j * (n + 1) + i; w = q + n + 1
                f += [(q, w, q + 1), (q + 1, w, w + 1)]
        return Mesh(v, f, self.sail_col)

    def build(self, built=True):
        hull = self._hull()
        det, guns = self._details(built)
        spars, ropes, sails = self._rig(built)
        out = {"hull": merge([hull, det] + ([self._rails()] if self.rails else [])), "deck": self._decks(),
               "spars": spars, "ropes": ropes, "sails": sails, "guns": guns}
        return out


# ── For the wrecks ───────────────────────────────────────────────────────────
def vnormals(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(m.v)
    for k in range(3):
        np.add.at(n, m.f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    return n / ln


def solidify(m, th, col=None):
    """Give a thin shell a thickness (planking seen edge-on where it broke): an inner copy and the edges closed."""
    n = vnormals(m)
    inner = Mesh(m.v - n * th, m.f[:, ::-1], m.c * 0.7)
    e = np.vstack([m.f[:, [0, 1]], m.f[:, [1, 2]], m.f[:, [2, 0]]])
    key = np.sort(e, 1)
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = e[cnt[inv.ravel()] == 1]
    N = len(m.v)
    side = (np.vstack([np.stack([bnd[:, 0], bnd[:, 1], bnd[:, 1] + N], 1), np.stack([bnd[:, 0], bnd[:, 1] + N, bnd[:, 0] + N], 1)])
            if len(bnd) else np.zeros((0, 3), int))
    c = np.vstack([m.c, inner.c])
    if col is not None and len(side):
        c[np.unique(side)] = col
    return Mesh(np.vstack([m.v, inner.v]), np.vstack([m.f, inner.f + N, side]), c)


def frames(ship, ts, col=(0.3, 0.22, 0.15), size=0.28, inset=0.25, upto=1.0):
    """The ribs at the stations ts (0 stern … 1 bow), each a square timber following the inside of the hull
    from the keel up to `upto` of the way to the rail; with the keelson along the middle."""
    P = []
    for t in ts:
        yb, yt = ship.keel(t), ship.keel(t) + (ship.top(t) - ship.keel(t)) * upto
        for s in (-1, 1):
            pts = []
            for u in np.linspace(0, 1, 9) ** 0.85:
                y = yb + 0.25 + (yt - yb) * u
                pts.append((s * max(0.0, float(ship.half(t, y)) - inset), y, ship.zt(t)))
            P.append(tube(pts, size / 2, col, seg=4))
    P.append(box(0.5, 0.5, ship.L * 0.85, col).T(0, ship.keel(0.5) + 0.2, 0))
    return merge(P)


def cannon(length=2.6, bore=0.09, bronze=True):
    """A muzzle-loading gun of the period, lying along +z with the breech at z = 0 (bronze or iron)."""
    col = (0.5, 0.42, 0.22) if bronze else (0.15, 0.14, 0.13)
    r = bore * 2.4
    prof = [(0.001, 0), (r * 1.2, 0.0), (r * 1.3, length * 0.04), (r * 1.4, length * 0.06), (r * 1.25, length * 0.08),
            (r * 1.15, length * 0.35), (r * 1.25, length * 0.37), (r * 1.05, length * 0.39), (r * 0.9, length * 0.9),
            (r * 1.05, length * 0.93), (r * 1.0, length * 0.97), (r * 0.95, length), (bore, length), (bore, length * 0.88),
            (0.001, length * 0.88)]
    m = lathe(prof, col, seg=12).R(math.pi / 2, 0, 0)
    knob = lathe([(0.001, 0), (r * 0.4, 0.02), (r * 0.5, 0.12), (r * 0.3, 0.2), (r * 0.4, 0.22), (0.001, 0.24)], col, seg=8)
    knob.R(-math.pi / 2, 0, 0)
    trun = cyl(r * 0.35, r * 0.35, r * 3.2, col, seg=6).R(0, 0, math.pi / 2).T(r * 1.6, 0, length * 0.42)
    g = merge([m, knob, trun])
    g.c = np.clip(g.c * (0.85 + 0.25 * fbm(g.v, 3.0, 9, 2))[:, None], 0, 1)
    return g


def gun_carriage(length=2.6, col=(0.36, 0.26, 0.17)):
    """A truck carriage under a gun of that length (the gun's trunnions sit on its cheeks)."""
    P = [box(0.12, 0.55, length * 0.5, col).T(s * 0.32, 0.2, length * 0.32) for s in (-1, 1)]
    P += [cyl(0.2, 0.2, 0.8, col, seg=8).R(0, 0, math.pi / 2).T(0.4, 0.2, length * 0.32 + dz) for dz in (-length * 0.2, length * 0.2)]
    P.append(box(0.7, 0.15, length * 0.5, col).T(0, 0.12, length * 0.32))
    return merge(P)


def cannonball(r=0.07):
    return lathe([(0.001, -r)] + [(r * math.sin(a), -r * math.cos(a)) for a in np.linspace(0.3, math.pi - 0.3, 6)] + [(0.001, r)],
                 (0.13, 0.12, 0.11), seg=8)


# ── Lettering and fittings ───────────────────────────────────────────────────
FONT5x7 = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    " ": ["....."] * 7,
}


def lettering(text, height, col, depth=0.04, arch=0.0):
    """Raised letters facing −z (a name on a stern), centred on x = 0, the baseline on y = 0; arch lifts the middle."""
    px = height / 7
    P = []
    n = len(text)
    width = (n * 6 - 1) * px
    for k, ch in enumerate(text.upper()):
        rows = FONT5x7[ch]
        x0 = -width / 2 + k * 6 * px
        for j, row in enumerate(rows):
            i = 0
            while i < 5:
                if row[i] == "#":
                    e = i
                    while e < 5 and row[e] == "#":
                        e += 1
                    x = x0 + (i + e) / 2 * px
                    y = (6 - j) * px + arch * (1 - (2 * x / max(width, 1e-6)) ** 2)
                    P.append(box((e - i) * px * 1.02, px * 1.02, depth, col, bottom=False).T(x, y + px / 2, -depth / 2))
                    i = e
                else:
                    i += 1
    m = merge(P)
    m.v[:, 0] *= -1                                  # read from behind (looking towards +z, +x is on the left)
    m.f = m.f[:, ::-1]
    return m


def star(r, col, depth=0.04):
    """A five-pointed star facing −z, centred on the origin."""
    v = [(0.0, 0.0, -depth)]
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.4
        v.append((rr * math.cos(a), rr * math.sin(a), -depth))
    f = [(0, 1 + (k + 1) % 10, 1 + k) for k in range(10)]
    return Mesh(v, f, col)


def wheel(r=0.75, col=(0.42, 0.28, 0.16), brass=(0.6, 0.48, 0.2)):
    """A ship's steering wheel on its pedestal: the rim, eight spokes with handles, the hub; the axle along z."""
    P = [tube([(r * math.cos(a), r * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 25)], 0.045, col, seg=5)]
    for k in range(8):
        a = k * math.pi / 4
        P.append(tube([(0, 0, 0), ((r + 0.22) * math.cos(a), (r + 0.22) * math.sin(a), 0)], 0.03, col, seg=4))
    P.append(cyl(0.13, 0.13, 0.16, brass, seg=10).R(math.pi / 2, 0, 0).T(0, 0, -0.08))
    P.append(cyl(0.06, 0.06, 0.6, brass, seg=6).R(math.pi / 2, 0, 0).T(0, 0, 0.0))
    ped = box(0.35, r + 0.35, 0.45, col).T(0, -(r + 0.35), 0.55)
    P.append(ped)
    return merge(P).T(0, r + 0.35, 0)
