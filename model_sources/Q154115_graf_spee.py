"""
Admiral Graf Spee (Q154115) — the wreck of the German "pocket battleship" off Montevideo, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Deutschland-class Panzerschiff completed in 1936: about 186 m long, beam 21.6 m, six 28 cm guns in two
triple turrets (fore and aft), the tower mast with the main rangefinder on top, one funnel with the aircraft
catapult behind it, 15 cm guns along the sides, diesel engines and two screws, the eagle on the stern.

The wreck, after the published descriptions: after the Battle of the River Plate she was scuttled by her own crew
off Montevideo on 17 December 1939 and burned for days. She settled upright in the soft bottom of the estuary in
about 11 m of muddy water; she was partially broken up in situ in 1942-43; in 2004 the main rangefinder and a
27-ton part of the command tower were raised, and the eagle from the stern was recovered before Uruguay stopped
further work. What is left is the hull deep in the mud with its turrets and the stumps of the superstructure,
the after part wrecked by the scuttling charges. The degree of burial, the heading (not published: 0) and what
the scrappers took are approximate.

    python model_sources/Q154115_graf_spee.py   -> models/item/Q154115/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm, yaw_quat  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1939)

L, B, D = 186.0, 21.6, 12.4
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.5, 0.07, 0.3, 1.6, 0.4, 2.4, 6.0
HULLC = (0.45, 0.47, 0.49)
SUPC = (0.55, 0.57, 0.58)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = [(0, 12), (50.5, 76), (113, L)]
HOUSES = [(52, 74, 0.72, 2.5, 0), (53, 64, 0.6, 2.5, 1), (53, 57.5, 0.6, 2.4, 2)]
FUNNELS = [(66.5, 1.7, 9.0, 0.85)]
MASTS = [(26.5, 16.0), (89.5, 15.0)]
BOATS = [(60.0, 1), (70.0, 1)]
BOAT_X = 0.32
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
RUDDER_D = L - 1.2
ANCHORS_GONE = ()
BS = dict(turrets=[(50.0, 1.0, 3, 0.0), (150.0, 1.0, 3, 3.1416)], gun_len=12.0, house_w=10.0,
          bridge=(62.0, 82.0, 4, 11.0), funnels=[(100.0, 2.8, 8.0, 0.75)],
          masts=[(118.0, 14.0, "pole")], casemates=[], secondary=[(d, s) for d in (88.0, 110.0, 122.0) for s in (-1, 1)], rudders=[0.0])


# ── Fittings ─────────────────────────────────────────────────────────────────


# ── Cargo ────────────────────────────────────────────────────────────────────


# ── The ship, piece by piece (driven by the description above) ───────────────
#   RAISED: [(d0, d1)] forecastle / islands / poop above the main deck, FC high
#   HOUSES: [(d0, d1, width fraction of B, height, level)] deck houses on the raised decks (level 0, 1, 2…)
#   BRIDGE_FRONT: d of the wheelhouse windows (on the top house) or None
#   FUNNELS: [(d, radius, height, squash)];  MASTS: [(d, height)];  BOATS: [(d, level)]
#   SCREWS: [(x, d, y, radius)];  RUDDER_D
LEVEL_H = 2.5


# ── Placing the wreck and things about it ────────────────────────────────────
CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45),
              (0.8, 0.45, 0.3), (0.42, 0.36, 0.55)]


# ── A battleship (driven by BS below) ───────────────────────────────────────
#   BS = dict(turrets=[(d, raised, guns, train)], gun_len, house_w, bridge=(d0, d1, levels, width), funnels=[(d, r, h, sx)],
#             masts=[(d, h, kind)], casemates=[d…] (both sides), secondary=[(d, side)], pagoda=None|(d, height))
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize")


SCREWS = [(-4.2, L - 14.0, 2.4, 2.1), (4.2, L - 14.0, 2.4, 2.1)]
RUDDER_D = L - 4.0


def tower():
    """The tower mast of the Panzerschiff: a slender armoured tower with platforms and the rangefinder on top."""
    P = [box(5.0, 18.0, 6.0, SUPC)]
    for k, y in enumerate((5.0, 9.0, 13.0)):
        P.append(box(9.0 - k, 0.3, 7.0 - k, SUPC).T(0, y, 0))
    P += [box(3.0, 2.0, 3.0, SUPC).T(0, 18.0, 0), box(10.5, 1.0, 1.4, GREY).T(0, 20.0, 0)]
    return merge(P)


wreck = merge(bs_part(False))
# broken up in 1942-43: the superstructure cut down, and nothing left within 1.5 m of the surface (11 m of water, the
# hull 3.5 m out of the mud): any face reaching above that goes, masts included (their tall faces have a low centre)
TOP = D + 6.0
cut = wreck.v[wreck.f][:, :, 1].max(axis=1) > TOP
wreck.f = wreck.f[~cut]
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.25, 0.1, 11)
aft = sstep(115, 128, dfb) * sstep(165, 150, dfb)                              # the after part wrecked by the scuttling charges and the fire
wreck.v[:, 1] -= aft * sstep(4.0, D + 3, wreck.v[:, 1]) * 4.0
crumple(wreck, 1.4, 0.2, 12, aft)
fc = wreck.v[wreck.f].mean(1)
wreck.f = wreck.f[~((aft[wreck.f].mean(1) > 0.5) & (fbm(fc, 0.12, 13) > 0.6))]
stump = merge([box(5.0, 3.0, 6.0, SUPC).T(0, D + 2.6, zd(72))]); crumple(stump, 0.4, 0.5, 14)   # on what the scrappers left of the bridge (top 1.5 m under the surface)
wreck = merge([wreck, stump])
wreck.T(0, -(D - 3.5), 0)                                                       # settled upright deep into the soft bottom
debris = merge(plates(40, 14, zd(140), 14, 100, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q154115" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-11, kind="warship", seed=1939)
ship = merge(bs_part(True) + [tower().T(0, deck_y(72) + 2.6 * 4, zd(72))]); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
