/* ═══════════════════════════════════════════════════════════════════════════
   Offlined Sea 3D Models — generic wreck generator                 (MIT licence)

   Draws a believable model for any catalog item that has no model of its own,
   from its family (liner, warship, submarine, cargo ship, sailing ship, aircraft,
   reef…) and its real length and beam. Same conventions as docs/MODEL_SPEC.md:
   metres, +Y up, bow to +Z, origin where it touches the seabed (y = 0 = seabed).

   Everything is deterministic: the same item always gives the same wreck
   (the random choices — list, burial, broken masts… — are seeded by its id).

       import * as THREE from "three";
       import { buildWreck } from "./sea3d-generator.js";
       const { mesh, info } = buildWreck(THREE, catalogItem);
       scene.add(mesh);

   The result is ONE mesh (one draw call) with vertex colours: paint, rust and
   sediment are baked in, so it needs no textures.
   ═══════════════════════════════════════════════════════════════════════════ */

export const GENERATOR_VERSION = "1.0";

// Typical size of each family when Wikidata has no length (m) and the usual length/beam ratio
const DEFAULTS = {
  liner:    { L: 200, ratio: 8.5 }, warship: { L: 150, ratio: 9 },  carrier: { L: 250, ratio: 7.6 },
  submarine:{ L: 70,  ratio: 11 },  cargo:   { L: 125, ratio: 7.2 }, tanker:  { L: 210, ratio: 6.6 },
  fishing:  { L: 28,  ratio: 3.8 }, steamer: { L: 95,  ratio: 7.4 }, ship:    { L: 80,  ratio: 7.2 },
  sail:     { L: 40,  ratio: 3.8 }, aircraft:{ L: 36,  ratio: 1 },
  reef:     { L: 70,  ratio: 1 },   vent:    { L: 50,  ratio: 1 },   site:    { L: 30,  ratio: 1 }, museum: { L: 40, ratio: 1 },
};
export const FAMILIES = Object.keys(DEFAULTS);

const C = {
  hull: 0x24272a, bottom: 0x6a2c20, white: 0xd6d2c4, grey: 0x6c7379, greyDark: 0x4a4f54, deck: 0x6a5a48,
  buff: 0xb48a50, black: 0x1c1c1c, wood: 0x4b3a2a, woodDark: 0x33281d, alu: 0xa9adb0, sub: 0x2a2d2f,
  rust: 0x5a3317, sediment: 0x8a8170, coral: [0xc2795a, 0xb39566, 0x8e6c9c, 0x7c9864, 0xd0b089, 0xa85a5a, 0x6f8fa0],
  terracotta: 0x985a3b, stone: 0x8d877c, concrete: 0x9a978e, chimneyW: 0xd8d3c6, chimneyB: 0x3a3330,
};

// ── Random numbers from the item id ─────────────────────────────────────────
function rng(seed) {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) { h = Math.imul(h ^ seed.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// Value noise (for rust and sediment patches)
function hash3(x, y, z) { let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 2147483647); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
function noise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  let r = 0;
  for (let k = 0; k < 8; k++) {
    const a = k & 1, b = (k >> 1) & 1, c = (k >> 2) & 1;
    r += hash3(xi + a, yi + b, zi + c) * (a ? u : 1 - u) * (b ? v : 1 - v) * (c ? w : 1 - w);
  }
  return r;
}

// ── Parts kit ───────────────────────────────────────────────────────────────
class Kit {
  constructor(T, rnd) { this.T = T; this.r = rnd; this.g = new T.Group(); }
  add(geo, color, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) {
    const m = new this.T.Mesh(geo);
    m.position.set(x, y, z); m.rotation.set(rx, ry, rz);
    if (typeof color === "function") m.userData.cf = color; else m.userData.c = color;
    this.g.add(m); return m;
  }
  /** Box standing on y (its bottom), centred on x/z */
  box(w, h, d, c, x, y, z, rx = 0, ry = 0, rz = 0) { return this.add(new this.T.BoxGeometry(w, h, d).translate(0, h / 2, 0), c, x, y, z, rx, ry, rz); }
  /** Vertical cylinder standing on y */
  cyl(r1, r2, h, c, x, y, z, seg = 14, rx = 0, ry = 0, rz = 0) { return this.add(new this.T.CylinderGeometry(r2, r1, h, seg).translate(0, h / 2, 0), c, x, y, z, rx, ry, rz); }
  /** Cylinder lying along Z, starting at z and going +len (rotate ry to aim it) */
  rod(r, len, c, x, y, z, ry = 0, rx = 0, seg = 8) {
    const g = new this.T.CylinderGeometry(r, r, len, seg).translate(0, len / 2, 0).rotateX(Math.PI / 2);
    return this.add(g, c, x, y, z, rx, ry, 0);
  }
  sphere(r, c, x, y, z, sx = 1, sy = 1, sz = 1, det = 1) { const g = new this.T.IcosahedronGeometry(r, det); g.scale(sx, sy, sz); return this.add(g, c, x, y, z); }
}

// ── Hull: lofted sections from stern (−L/2) to bow (+L/2) ───────────────────
function hull(T, L, B, H, o) {
  const NS = 56, NP = 9, st = [];
  for (let i = 0; i <= NS; i++) {
    const t = i / NS, z = -L / 2 + t * L;
    let w = 1;
    if (t > 1 - o.bow) w = Math.pow(Math.sin(((1 - t) / o.bow) * Math.PI / 2), o.bowPow ?? 0.9);
    if (t < o.stern) w = Math.min(w, o.transom + (1 - o.transom) * Math.sin((t / o.stern) * Math.PI / 2));
    w = Math.max(w, 0.006);
    let yb = 0;
    const fb = o.bow * 0.7;
    if (t > 1 - fb) { const u = (t - (1 - fb)) / fb; yb = H * o.rake * u * u; }
    if (t < o.stern * 0.8) { const u = 1 - t / (o.stern * 0.8); yb = Math.max(yb, H * (o.sternRise ?? 0.4) * u * u); }
    const yd = H * (1 + o.sheer * Math.pow(Math.abs(2 * t - 1), 2) + (t > 0.5 ? (o.bowRise || 0) * Math.pow(2 * t - 1, 3) : 0));
    st.push({ z, w, yb, yd });
  }
  const pos = [], idx = [], W = 2 * NP + 1;
  for (const s of st) for (let j = 0; j < W; j++) {
    const side = j < NP ? -1 : 1, u = Math.abs(NP - j) / NP;
    pos.push(side * (B / 2) * s.w * (1 - Math.pow(1 - u, o.k)), s.yb + (s.yd - s.yb) * u, s.z);
  }
  for (let i = 0; i < NS; i++) for (let j = 0; j < W - 1; j++) {
    const a = i * W + j, b = a + W, c = a + 1, d = b + 1;
    idx.push(a, c, b, b, c, d);
  }
  // deck
  const d0 = pos.length / 3;
  for (const s of st) pos.push(-(B / 2) * s.w, s.yd, s.z, (B / 2) * s.w, s.yd, s.z);
  for (let i = 0; i < NS; i++) { const a = d0 + i * 2, b = a + 2, c = a + 1, d = b + 1; idx.push(a, b, c, c, b, d); }
  // transom
  const c0 = pos.length / 3, s0 = st[0];
  pos.push(0, (s0.yb + s0.yd) / 2, s0.z);
  for (let j = 0; j < W - 1; j++) idx.push(c0, j + 1, j);
  const g = new T.BufferGeometry();
  g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  const deckAt = z => {
    const t = clamp((z + L / 2) / L, 0, 1) * NS, i = Math.min(NS - 1, Math.floor(t)), f = t - i;
    return st[i].yd * (1 - f) + st[i + 1].yd * f;
  };
  const halfAt = z => {
    const t = clamp((z + L / 2) / L, 0, 1) * NS, i = Math.min(NS - 1, Math.floor(t)), f = t - i;
    return (B / 2) * (st[i].w * (1 - f) + st[i + 1].w * f);
  };
  return { geo: g, deckAt, halfAt };
}

/** Body of revolution along Z: prof(t) gives the radius at t∈[0,1] from tail to nose */
function lathe(T, len, prof, n = 36, seg = 18) {
  const pts = [];
  for (let i = 0; i <= n; i++) { const t = i / n; pts.push(new T.Vector2(Math.max(0.001, prof(t)), -len / 2 + t * len)); }
  return new T.LatheGeometry(pts, seg).rotateX(Math.PI / 2);
}

// ── Families ────────────────────────────────────────────────────────────────
function vessel(K, fam, L, B, built) {
  const T = K.T, r = K.r, modern = built ? built >= 1955 : fam === "tanker";
  const P = {
    liner:   { k: 4,   bow: 0.18, stern: 0.12, transom: 0.35, sheer: 0.06, rake: 0.6, bowRise: 0.05, H: 0.8 },
    warship: { k: 2.6, bow: 0.26, bowPow: 1.1, stern: 0.15, transom: 0.55, sheer: 0.04, rake: 0.3, bowRise: 0.08, H: 0.62 },
    carrier: { k: 3.5, bow: 0.2, stern: 0.12, transom: 0.6, sheer: 0.02, rake: 0.5, H: 0.72 },
    cargo:   { k: 6,   bow: 0.15, stern: 0.1, transom: 0.4, sheer: 0.05, rake: 0.55, H: 0.62 },
    tanker:  { k: 7,   bow: 0.12, stern: 0.08, transom: 0.55, sheer: 0.02, rake: 0.6, H: 0.6 },
    steamer: { k: 4.5, bow: 0.16, stern: 0.12, transom: 0.35, sheer: 0.07, rake: 0.5, H: 0.62 },
    ship:    { k: 4.5, bow: 0.16, stern: 0.12, transom: 0.4, sheer: 0.06, rake: 0.5, H: 0.62 },
    fishing: { k: 3,   bow: 0.25, stern: 0.15, transom: 0.6, sheer: 0.12, rake: 0.45, bowRise: 0.08, H: 0.55 },
    sail:    { k: 2.2, bow: 0.22, stern: 0.12, transom: 0.65, sheer: 0.22, rake: 0.35, bowRise: 0.12, sternRise: 0.3, H: 0.72 },
  }[fam] || {};
  const H = B * P.H;
  const war = fam === "warship" || fam === "carrier";
  const top = war ? C.grey : fam === "sail" ? C.wood : C.hull;
  const water = H * 0.55;
  const hullColor = fam === "sail" ? C.wood : (p => (p[1] < water ? C.bottom : top));
  const h = hull(T, L, B, H, P);
  K.add(h.geo, hullColor);
  const D = h.deckAt, sup = war ? C.grey : C.white, u = Math.max(2.4, B * 0.11);   // u: one deck height

  if (fam === "liner") {
    const n = clamp(Math.round(L / 55), 2, 5);
    for (let i = 0; i < n; i++) {
      const len = L * (0.6 - i * 0.09), w = B * (0.9 - i * 0.06), z = -0.03 * L;
      K.box(w, u, len, sup, 0, D(z) + i * u - 0.2, z);
    }
    const nf = built && built < 1914 && L > 200 ? 4 : built && built < 1932 ? (L > 170 ? 3 : 2) : L > 150 ? 2 : 1;
    const ty = D(0) + n * u - 0.2;
    for (let i = 0; i < nf; i++) {
      const z = nf === 1 ? -0.02 * L : L * (0.17 - (0.34 * i) / (nf - 1));
      const fr = B * 0.085, fh = B * 0.75;
      K.cyl(fr, fr * 0.95, fh * 0.8, C.buff, 0, ty, z, 16, -0.12).scale.set(0.8, 1, 1.3);
      K.cyl(fr * 0.95, fr * 0.95, fh * 0.2, C.black, 0, ty + fh * 0.8 * Math.cos(0.12), z - fh * 0.8 * Math.sin(0.12), 16, -0.12).scale.set(0.8, 1, 1.3);
    }
    for (const z of [0.38, -0.36]) K.cyl(0.35, 0.25, B * 1.5, C.alu, 0, D(z * L), z * L, 6, -0.08);
    const nb = Math.floor(L * 0.4 / 14);
    for (let i = 0; i < nb; i++) for (const s of [-1, 1]) K.box(2.2, 1.2, 8, C.white, s * (B * (0.9 - (n - 1) * 0.06) / 2 - 1.4), ty, L * 0.15 - i * 14);
  }

  if (fam === "warship") {
    const big = L >= 170, mid = L >= 100;
    K.box(B * 0.55, u * 2, L * 0.13, sup, 0, D(0.17 * L), 0.17 * L);                // bridge
    K.box(B * 0.38, u * 1.6, L * 0.07, sup, 0, D(0.17 * L) + u * 2, 0.185 * L);
    K.cyl(0.6, 0.4, B * 1.4, sup, 0, D(0.15 * L) + u * 3.6, 0.15 * L, 6);              // mast
    const nf = big ? 2 : 1;
    for (let i = 0; i < nf; i++) K.cyl(B * 0.08, B * 0.075, B * 0.6, C.greyDark, 0, D(0), (0.03 - i * 0.11) * L, 14, -0.05).scale.set(0.85, 1, 1.4);
    K.box(B * 0.45, u * 1.2, L * 0.1, sup, 0, D(-0.17 * L), -0.17 * L);
    const tur = (z, up, aft, s) => {
      const y = D(z) + up, rb = B * 0.16 * s;
      K.cyl(rb, rb, B * 0.1, C.greyDark, 0, y, z, 18);
      K.box(B * 0.3 * s, B * 0.13 * s, B * 0.34 * s, C.grey, 0, y + B * 0.1, z);
      const nb = big ? 3 : 2;
      for (let i = 0; i < nb; i++) K.rod(B * 0.017 * s, B * 0.8 * s, C.greyDark, (i - (nb - 1) / 2) * B * 0.08 * s, y + B * 0.16 * s, z + (aft ? -1 : 1) * B * 0.12 * s, aft ? Math.PI : 0, aft ? 0.06 : -0.06);
    };
    if (big) { tur(0.34 * L, 0, false, 1); tur(0.25 * L, B * 0.18, false, 1); tur(-0.27 * L, B * 0.18, true, 1); tur(-0.36 * L, 0, true, 1); }
    else if (mid) { tur(0.3 * L, 0, false, 0.8); tur(-0.32 * L, 0, true, 0.8); }
    else {
      tur(0.3 * L, 0, false, 0.6); tur(-0.34 * L, 0, true, 0.6);
      for (const s of [-1, 1]) K.rod(0.3, L * 0.05, C.greyDark, s * B * 0.25, D(-0.08 * L) + 0.4, -0.1 * L, s * 0.5);   // torpedo tubes
    }
  }

  if (fam === "carrier") {
    const fy = D(0) + u * 0.6;
    K.box(B * 0.6, u * 0.6, L * 0.9, C.grey, 0, D(0), 0);                              // hangar sides
    K.box(B * 1.18, 1.4, L * 1.0, C.greyDark, 0, fy, 0.01 * L);                         // flight deck
    const ix = B * 0.46, iz = 0.08 * L;
    K.box(B * 0.12, u * 4, L * 0.13, C.grey, ix, fy + 1.4, iz);                         // island
    K.box(B * 0.1, u * 1.6, L * 0.05, C.greyDark, ix, fy + 1.4 + u * 4, iz - 0.02 * L);
    K.cyl(0.5, 0.3, B * 0.7, C.grey, ix, fy + 1.4 + u * 4, iz + 0.03 * L, 6);
  }

  if (fam === "cargo" || fam === "tanker" || fam === "steamer" || fam === "ship") {
    if (modern || fam === "tanker") {
      const z = -0.38 * L, n = clamp(Math.round(B / 6), 3, 6);
      for (let i = 0; i < n; i++) K.box(B * (0.92 - i * 0.07), u, L * (0.12 - i * 0.012), C.white, 0, D(z) + i * u, z);
      K.cyl(B * 0.07, B * 0.07, B * 0.4, C.black, 0, D(z) + n * u, z - L * 0.04, 12).scale.set(1, 1, 1.4);
      if (fam === "tanker") {
        K.rod(0.5, L * 0.6, C.greyDark, 0, D(0) + 1.2, -0.3 * L);
        K.rod(0.35, L * 0.6, C.greyDark, B * 0.18, D(0) + 0.8, -0.3 * L);
        K.box(B * 0.05, 0.8, L * 0.6, C.grey, -B * 0.15, D(0) + 1.8, 0);
      } else {
        const nh = clamp(Math.round(L / 30), 3, 7);
        for (let i = 0; i < nh; i++) K.box(B * 0.72, 1.6, L * 0.07, C.greyDark, 0, D(0), L * (0.36 - (0.62 * i) / Math.max(1, nh - 1)));
        for (let i = 0; i < nh - 1; i++) K.cyl(0.9, 0.7, B * 0.7, C.buff, B * 0.3 * (i % 2 ? 1 : -1), D(0), L * (0.31 - (0.62 * i) / Math.max(1, nh - 1)), 8);
      }
    } else {
      K.box(B * 0.62, u * 2, L * 0.11, C.white, 0, D(0.06 * L), 0.06 * L);              // midship bridge
      K.box(B * 0.4, u, L * 0.06, C.white, 0, D(0.06 * L) + u * 2, 0.07 * L);
      K.cyl(B * 0.07, B * 0.065, B * 0.9, C.black, 0, D(0), -0.02 * L, 12, -0.1);
      for (const z of [0.3, -0.28]) {
        K.cyl(0.4, 0.25, B * 1.5 * (0.6 + 0.4 * r()), C.wood, 0, D(z * L), z * L, 6, -0.05);
        K.rod(0.18, B * 0.8, C.wood, 0, D(z * L) + 2, z * L, z > 0 ? 0 : Math.PI, z > 0 ? -0.5 : 0.5);  // derrick
      }
      for (const z of [0.22, 0.14, -0.18, -0.3]) K.box(B * 0.45, 0.9, L * 0.06, C.deck, 0, D(z * L), z * L);
    }
  }

  if (fam === "fishing") {
    K.box(B * 0.62, u * 1.3, L * 0.2, C.white, 0, D(0.12 * L), 0.12 * L);
    K.cyl(0.18, 0.12, B * 1.3, C.alu, 0, D(0.12 * L) + u * 1.3, 0.14 * L, 6);
    for (const s of [-1, 1]) K.cyl(0.25, 0.25, B * 0.8, C.buff, s * B * 0.38, D(-0.42 * L), -0.42 * L, 6);
    K.rod(0.22, B * 0.76, C.buff, -B * 0.38, D(-0.42 * L) + B * 0.8, -0.42 * L, Math.PI / 2);
  }

  if (fam === "sail") {
    const ac = L * 0.24;
    K.box(B * 0.8, B * 0.32, ac, C.woodDark, 0, D(-0.38 * L + ac / 2) - 0.3, -0.38 * L + ac / 2);
    if (!built || built < 1720) K.box(B * 0.62, B * 0.22, L * 0.14, C.woodDark, 0, D(0.33 * L) - 0.3, 0.33 * L);
    const nm = L > 32 ? 3 : 2, ms = [0.24, 0.02, -0.22];
    for (let i = 0; i < nm; i++) {
      const z = ms[i] * L, mh = B * 2.4 * (0.12 + 0.45 * r());                          // broken masts
      K.cyl(B * 0.045, B * 0.04, mh, C.woodDark, 0, D(z), z, 7, (r() - 0.5) * 0.3, 0, (r() - 0.5) * 0.3);
    }
    K.rod(B * 0.035, L * 0.18, C.woodDark, 0, D(0.5 * L) - 0.3, 0.45 * L, 0, -0.35);        // bowsprit
  }
  return { H };
}

function submarine(K, L, B, built) {
  const R = B / 2, modern = built ? built >= 1955 : false;
  if (modern) {
    const prof = t => R * (t > 0.86 ? Math.sqrt(Math.max(0, 1 - ((t - 0.86) / 0.14) ** 2)) : t < 0.32 ? Math.pow(t / 0.32, 0.75) : 1);
    K.add(lathe(K.T, L, prof), C.sub, 0, R, 0);
    K.box(R * 0.55, R * 1.5, L * 0.1, C.sub, 0, R * 1.8, 0.2 * L);
    K.box(R * 2.4, R * 0.08, L * 0.035, C.sub, 0, R * 2.7, 0.21 * L);                    // sail planes
    K.box(R * 2.6, R * 0.08, L * 0.05, C.sub, 0, R, -0.43 * L);                           // tail fins
    K.box(R * 0.08, R * 2.6, L * 0.05, C.sub, 0, -0.3 * R, -0.43 * L);
  } else {
    const r0 = R * 0.78;
    const prof = t => r0 * (t > 0.82 ? Math.pow(Math.max(0, (1 - t) / 0.18), 0.6) : t < 0.25 ? Math.pow(t / 0.25, 0.7) : 1);
    K.add(lathe(K.T, L, prof), C.sub, 0, r0, 0);
    const prof2 = t => R * 0.95 * (t > 0.75 ? Math.pow(Math.max(0, (1 - t) / 0.25), 0.8) : t < 0.2 ? Math.pow(t / 0.2, 0.8) : 1);
    K.add(lathe(K.T, L * 0.6, prof2), C.sub, 0, r0 * 0.9, 0.02 * L);                      // saddle tanks
    K.box(R * 0.7, R * 0.4, L * 0.7, C.greyDark, 0, r0 * 1.62, 0.04 * L);                 // deck casing
    K.box(R * 0.4, R * 0.3, L * 0.1, C.greyDark, 0, r0 * 1.45, 0.43 * L, 0.25);            // bow casing sloping down
    K.box(R * 0.7, R * 1.2, L * 0.075, C.sub, 0, r0 * 1.7 + R * 0.4, 0.04 * L);           // conning tower
    K.cyl(R * 0.35, R * 0.35, R * 1.2, C.sub, 0, r0 * 1.7 + R * 0.4, 0.04 * L + L * 0.037, 12);
    K.cyl(0.25, 0.25, R * 0.4, C.greyDark, 0, r0 * 1.7 + R * 0.45, 0.15 * L, 8);          // deck gun
    K.rod(0.09, R * 1.3, C.greyDark, 0, r0 * 1.7 + R * 0.75, 0.15 * L, 0, -0.1);
    K.cyl(0.08, 0.06, R * 2.2, C.alu, 0, r0 * 1.7 + R * 1.5, 0.05 * L, 5);              // periscope
  }
  return { H: B };
}

function aircraft(K, L, rnd) {
  const T = K.T, small = L < 16, R = L * (small ? 0.065 : 0.05), span = L * (small ? 1.2 : 0.95);
  const prof = t => R * (t > 0.88 ? Math.sqrt(Math.max(0, 1 - ((t - 0.88) / 0.12) ** 2)) : t < 0.28 ? 0.25 + 0.75 * Math.pow(t / 0.28, 0.8) : 1);
  K.add(lathe(T, L, prof, 30, 14), C.alu, 0, R, 0);
  const wing = (side, chord, half, sweep, y, z, th) => {
    const s = new T.Shape();
    s.moveTo(0, 0); s.lineTo(side * half, sweep); s.lineTo(side * half, sweep + chord * 0.35); s.lineTo(0, chord);
    const g = new T.ExtrudeGeometry(s, { depth: th, bevelEnabled: false }).rotateX(-Math.PI / 2);
    return K.add(g, C.alu, 0, y, z);
  };
  const c0 = L * (small ? 0.2 : 0.17), wz = small ? 0.12 * L : 0.06 * L;
  const broken = rnd() < 0.4;                                                             // one wing broken off
  wing(1, c0, span / 2 * (broken ? 0.45 : 1), small ? c0 * 0.15 : span * 0.15, R * 0.35, wz, R * 0.25);
  wing(-1, c0, span / 2, small ? c0 * 0.15 : span * 0.15, R * 0.35, wz, R * 0.25);
  if (broken) wing(1, c0 * 0.6, span * 0.25, 0, 0.1, wz - span * 0.12, R * 0.2).position.x = span * 0.4;
  const tz = -0.42 * L;
  wing(1, L * 0.09, span * 0.17, L * 0.04, R * 1.1, tz + L * 0.07, R * 0.15);
  wing(-1, L * 0.09, span * 0.17, L * 0.04, R * 1.1, tz + L * 0.07, R * 0.15);
  const f = new T.Shape(); f.moveTo(0, 0); f.lineTo(-L * 0.08, L * 0.11); f.lineTo(-L * 0.13, L * 0.11); f.lineTo(-L * 0.12, 0);
  K.add(new T.ExtrudeGeometry(f, { depth: R * 0.15, bevelEnabled: false }).rotateY(-Math.PI / 2), C.alu, R * 0.07, R * 1.6, tz + L * 0.14);
  if (small) K.cyl(R * 0.15, R * 0.15, L * 0.05, C.black, 0, R, L * 0.5, 8, Math.PI / 2);
  else for (const s of [-1, 1]) K.add(lathe(T, L * 0.12, t => R * 0.42 * (t < 0.2 ? 0.7 + 1.5 * t : 1)), C.greyDark, s * span * 0.17, R * 0.1, wz + c0 * 0.2);
  return { H: R * 2 };
}

function reef(K, L, r) {
  const n = 70 + Math.floor(r() * 60), R = L / 2;
  for (let i = 0; i < n; i++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * R, x = Math.cos(a) * d, z = Math.sin(a) * d;
    const s = 0.8 + r() * 3 * (1 - d / R * 0.6), c = C.coral[Math.floor(r() * C.coral.length)];
    if (r() < 0.65) K.sphere(s, c, x, s * 0.3, z, 1 + r() * 0.6, 0.5 + r() * 0.7, 1 + r() * 0.6);
    else for (let b = 0; b < 4; b++) K.cyl(s * 0.12, s * 0.08, s * (1 + r()), c, x + (r() - 0.5) * s, 0, z + (r() - 0.5) * s, 5, (r() - 0.5) * 0.7, 0, (r() - 0.5) * 0.7);
  }
  return { H: 4 };
}
function vent(K, L, r, white) {
  const n = 6 + Math.floor(r() * 8), col = white ? C.chimneyW : C.chimneyB;
  for (let i = 0; i < n; i++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * L * 0.45, x = Math.cos(a) * d, z = Math.sin(a) * d;
    let h = (i === 0 ? 0.9 : 0.15 + r() * 0.5) * L * 0.8, y = 0, rad = h * 0.09 + 0.8;
    while (h > 0.5) { const seg = Math.min(h, 3 + r() * 6); K.cyl(rad, rad * 0.8, seg, col, x + (r() - 0.5) * 0.4, y, z + (r() - 0.5) * 0.4, 9); y += seg * 0.97; h -= seg; rad *= 0.8; }
    for (let k = 0; k < 3; k++) K.sphere(rad * 1.6, col, x + (r() - 0.5) * 2, y * r(), z + (r() - 0.5) * 2, 1.2, 0.5, 1.2, 0);
  }
  return { H: L * 0.8 };
}
function site(K, L, r) {
  const amph = t => 0.32 * (t < 0.12 ? 0.3 + t * 3 : t < 0.6 ? Math.sin(Math.PI * (t - 0.05) / 1.1) : t < 0.85 ? 0.6 - (t - 0.6) : 0.25);
  const n = 50 + Math.floor(r() * 40);
  for (let i = 0; i < n; i++) {
    const x = (r() - 0.5) * L * 0.8, z = (r() - 0.5) * L * 0.5;
    const m = K.add(lathe(K.T, 1.1, amph, 12, 10), C.terracotta, x, 0.15, z, (r() - 0.5) * 0.6, r() * 6, 0);
    if (r() < 0.2) m.rotation.x += Math.PI / 2;                                           // a few still standing
  }
  for (let i = 0; i < 12; i++) K.sphere(0.5 + r() * 1.2, C.stone, (r() - 0.5) * L, 0, (r() - 0.5) * L, 1.3, 0.6, 1, 0);
  return { H: 1 };
}
function museum(K, L, r) {
  const n = 18 + Math.floor(r() * 18);
  for (let i = 0; i < n; i++) {
    const x = ((i % 6) - 2.5) * 3.2 + (r() - 0.5), z = (Math.floor(i / 6) - 2) * 3.5 + (r() - 0.5), ry = r() * 6;
    K.cyl(0.24, 0.2, 1.05, C.concrete, x - 0.15, 0, z, 7, 0, ry); K.cyl(0.24, 0.2, 1.05, C.concrete, x + 0.15, 0, z, 7, 0, ry);
    K.cyl(0.3, 0.38, 0.7, C.concrete, x, 0.95, z, 8, 0, ry);
    K.sphere(0.13, C.concrete, x, 1.85, z, 1, 1.2, 1, 1);
  }
  return { H: 2 };
}

// ── Weathering, merging ────────────────────────────────────────────────────
const _c = { r: 0, g: 0, b: 0 };
function hexToLinear(T, hex, out) { const c = new T.Color(hex); out.r = c.r; out.g = c.g; out.b = c.b; return out; }

function finish(T, K, o) {
  const pos = [], nor = [], col = [], base = {}, rust = hexToLinear(T, C.rust, {}), sed = hexToLinear(T, C.sediment, {});
  K.g.updateMatrixWorld(true);
  const nm = new T.Matrix3(), v = new T.Vector3(), n = new T.Vector3();
  K.g.traverse(m => {
    if (!m.isMesh) return;
    let g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
    if (!g.attributes.normal) g.computeVertexNormals();
    const P = g.attributes.position, N = g.attributes.normal;
    nm.getNormalMatrix(m.matrixWorld);
    for (let i = 0; i < P.count; i++) {
      const hex = m.userData.cf ? m.userData.cf([P.getX(i), P.getY(i), P.getZ(i)]) : m.userData.c;
      const b = base[hex] || (base[hex] = hexToLinear(T, hex, {}));
      v.fromBufferAttribute(P, i).applyMatrix4(m.matrixWorld);
      n.fromBufferAttribute(N, i).applyMatrix3(nm).normalize();
      pos.push(v.x, v.y, v.z); nor.push(n.x, n.y, n.z);
      const ru = o.rust * sstep(0.35, 0.75, noise3(v.x * 0.09, v.y * 0.09, v.z * 0.09) + noise3(v.x * 0.5, v.y * 0.5, v.z * 0.5) * 0.25);
      const se = o.sed * sstep(0.45, 0.95, Math.abs(n.y)) * (n.y > 0 ? 1 : 0.3) * (0.6 + 0.4 * noise3(v.x * 0.2, 0, v.z * 0.2));
      const low = sstep(1.5, 0, v.y) * 0.5;                               // seabed silt up the sides
      let r = b.r + (rust.r - b.r) * ru, gg = b.g + (rust.g - b.g) * ru, bb = b.b + (rust.b - b.b) * ru;
      const k = Math.min(1, se + low);
      r += (sed.r - r) * k; gg += (sed.g - gg) * k; bb += (sed.b - bb) * k;
      col.push(r, gg, bb);
    }
    g.dispose(); m.geometry.dispose();
  });
  const geo = new T.BufferGeometry();
  geo.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
  geo.setAttribute("normal", new T.Float32BufferAttribute(nor, 3));
  geo.setAttribute("color", new T.Float32BufferAttribute(col, 3));
  geo.computeBoundingBox(); geo.computeBoundingSphere();
  return geo;
}

/**
 * The generic model of a catalog item.
 * item: an entry of catalog.json ({ q, fam, length?, beam?, built?, name? … }).
 * Returns { mesh, info: { fam, length, beam, generic: true, version } }.
 */
export function buildWreck(T, item, opts = {}) {
  const fam = DEFAULTS[item.fam] ? item.fam : "ship", d = DEFAULTS[fam];
  const r = rng(String(item.q || item.name || "x"));
  const place = fam === "reef" || fam === "vent" || fam === "site" || fam === "museum";   // a patch of a place, not an object: fixed size
  let L = !place && item.length && item.length >= 4 && item.length <= 480 ? item.length : d.L;
  let B = item.beam && item.beam > L / 25 && item.beam < L * 0.6 ? item.beam : L / d.ratio;
  if (fam === "aircraft" && !item.length) L = 30 + r() * 30;
  const K = new Kit(T, r), built = item.built;
  let H, burial, roll, pitch, rustAmt, sedAmt;
  if (fam === "submarine") { ({ H } = submarine(K, L, B, built)); burial = 0.1 + r() * 0.2; roll = (r() - 0.5) * 0.5; pitch = (r() - 0.5) * 0.06; rustAmt = 0.45; sedAmt = 0.45; }
  else if (fam === "aircraft") { ({ H } = aircraft(K, L, r)); burial = 0.15 + r() * 0.25; roll = (r() - 0.5) * 0.3; pitch = (r() - 0.5) * 0.05; rustAmt = 0.15; sedAmt = 0.55; }
  else if (fam === "reef") { ({ H } = reef(K, L, r)); burial = 0; roll = 0; pitch = 0; rustAmt = 0; sedAmt = 0.1; }
  else if (fam === "vent") { ({ H } = vent(K, L, r, /lost city/i.test(item.name || "") || r() < 0.3)); burial = 0; roll = 0; pitch = 0; rustAmt = 0; sedAmt = 0.15; }
  else if (fam === "site") { ({ H } = site(K, L, r)); burial = 0; roll = 0; pitch = 0; rustAmt = 0; sedAmt = 0.5; }
  else if (fam === "museum") { ({ H } = museum(K, L, r)); burial = 0; roll = 0; pitch = 0; rustAmt = 0; sedAmt = 0.25; }
  else {
    ({ H } = vessel(K, fam, L, B, built));
    const old = fam === "sail" || (built && built < 1900);
    burial = old ? 0.3 + r() * 0.25 : 0.08 + r() * 0.17;
    roll = (r() < 0.5 ? -1 : 1) * (0.03 + r() * (old ? 0.4 : 0.3)); pitch = (r() - 0.5) * 0.05;
    rustAmt = fam === "sail" ? 0.1 : old ? 0.75 : 0.5; sedAmt = old ? 0.8 : 0.5;
    if (item.sunk && item.sunk > 1990) { rustAmt *= 0.5; sedAmt *= 0.5; }
  }
  if (opts.clean) { rustAmt = 0; sedAmt = 0; burial = 0; roll = 0; pitch = 0; }
  K.g.rotation.set(pitch, 0, roll);
  const geo = finish(T, K, { rust: rustAmt, sed: sedAmt });
  geo.translate(0, -geo.boundingBox.min.y - burial * H, 0);                 // lowest point buried below the seabed (y = 0)
  geo.computeBoundingBox(); geo.computeBoundingSphere();
  const mat = opts.material || new T.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0.05, side: T.DoubleSide });
  const mesh = new T.Mesh(geo, mat);
  mesh.name = item.name || item.q || fam;
  return { mesh, info: { fam, length: L, beam: B, generic: true, version: GENERATOR_VERSION, triangles: geo.attributes.position.count / 3 } };
}
