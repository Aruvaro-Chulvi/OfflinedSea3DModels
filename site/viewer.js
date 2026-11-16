// Offlined Sea 3D Models — catalog viewer (MIT)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { buildWreck, GENERATOR_VERSION } from "./generator/sea3d-generator.js";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const REPO = "https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels";

// ── Texts ────────────────────────────────────────────────────────────────────
const TX = {
  en: {
    sub: "{n} wrecks and places under the sea with a Wikipedia article · {m} with their own 3D model",
    contrib: "How to help", search: "Search a wreck, a ship, a place…", allFam: "All families", allLang: "Any language",
    wl: "Article in {l}", own: "With its own model", count: "{n} found", more: "Show more", empty: "Choose a wreck on the list",
    length: "Length", beam: "Beam", depth: "Depth", built: "Built", sunk: "Lost", pos: "Position", klass: "Class", vehicle: "Aircraft",
    wp: "Wikipedia", wd: "Wikidata", ownM: "Own model by {a} · {lic}", classM: "Model of its class, by {a} · {lic}",
    gen: "Generic model, drawn by code from its family and size (generator {v}). <a href=\"{u}\">Make a better one!</a>",
    glb: "⬇ Download .glb", asBuilt: "As built", asWreck: "As a wreck", diver: "The figure beside it is a 1.8 m diver.",
    approx: "Depth from the GEBCO grid (~1 km cells): coastal wrecks may be shallower or deeper.", m: "m", coast: "near the coast",
    fam: { liner: "Liner or ferry", warship: "Warship", carrier: "Aircraft carrier", submarine: "Submarine", cargo: "Cargo ship", tanker: "Tanker",
           fishing: "Fishing vessel", steamer: "Steamship", ship: "Ship", sail: "Sailing ship", aircraft: "Aircraft", reef: "Reef",
           vent: "Hydrothermal vent", site: "Archaeological site", museum: "Underwater museum" },
  },
  es: {
    sub: "{n} pecios y lugares bajo el mar con artículo en Wikipedia · {m} con modelo 3D propio",
    contrib: "Cómo ayudar", search: "Busca un pecio, un barco, un lugar…", allFam: "Todas las familias", allLang: "Cualquier idioma",
    wl: "Artículo en {l}", own: "Con modelo propio", count: "{n} encontrados", more: "Ver más", empty: "Elige un pecio de la lista",
    length: "Eslora", beam: "Manga", depth: "Profundidad", built: "Construido", sunk: "Perdido", pos: "Posición", klass: "Clase", vehicle: "Avión",
    wp: "Wikipedia", wd: "Wikidata", ownM: "Modelo propio de {a} · {lic}", classM: "Modelo de su clase, de {a} · {lic}",
    gen: "Modelo genérico, dibujado por código según su familia y tamaño (generador {v}). <a href=\"{u}\">¡Haz uno mejor!</a>",
    glb: "⬇ Descargar .glb", asBuilt: "Como era", asWreck: "Como pecio", diver: "La figura de al lado es un buzo de 1,8 m.",
    approx: "Profundidad de la malla de GEBCO (celdas de ~1 km): en la costa el pecio puede estar más o menos hondo.", m: "m", coast: "cerca de la costa",
    fam: { liner: "Transatlántico o ferri", warship: "Buque de guerra", carrier: "Portaaviones", submarine: "Submarino", cargo: "Carguero", tanker: "Petrolero",
           fishing: "Pesquero", steamer: "Vapor", ship: "Barco", sail: "Velero", aircraft: "Avión", reef: "Arrecife",
           vent: "Fuente hidrotermal", site: "Yacimiento arqueológico", museum: "Museo submarino" },
  },
};
const ICON = { liner: "🛳️", warship: "⚓", carrier: "🛩️", submarine: "🐋", cargo: "🚢", tanker: "🛢️", fishing: "🎣", steamer: "🚢", ship: "🚢",
               sail: "⛵", aircraft: "✈️", reef: "🪸", vent: "🌋", site: "🏺", museum: "🗿" };
const LANGS = { en: "English", es: "Español", de: "Deutsch", fr: "Français", it: "Italiano", pt: "Português", ru: "Русский", zh: "中文", ar: "العربية", hi: "हिन्दी" };
let L = (() => { try { const s = localStorage.getItem("s3d_lang"); if (s) return s; } catch { /* */ } return /^es\b/i.test(navigator.language) ? "es" : "en"; })();
const t = (k, v = {}) => String(TX[L][k] ?? TX.en[k] ?? k).replace(/\{(\w+)\}/g, (_, x) => v[x] ?? "");
const fmt = n => Number(n).toLocaleString(L === "es" ? "es-ES" : "en-US", { maximumFractionDigits: 1 });

// ── Data ─────────────────────────────────────────────────────────────────────
let CAT = [], IDX = { item: {}, class: {}, vehicle: {} }, shown = 300, cur = null, clean = false;
const ownOf = it => IDX.item[it.q] ? ["item", IDX.item[it.q]] : it.class && IDX.class[it.class] ? ["class", IDX.class[it.class]]
  : it.vehicle && IDX.vehicle[it.vehicle] ? ["vehicle", IDX.vehicle[it.vehicle]] : null;
const norm = s => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

async function load() {
  const [c, i] = await Promise.all([
    fetch("catalog.json").then(r => r.json()),
    fetch("models/index.json").then(r => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  if (i) IDX = i;
  CAT = c.items;
  for (const it of CAT) it._s = norm(it.name + " " + Object.values(it.titles).join(" ") + " " + it.q);
  CAT.sort((a, b) => (ownOf(b) ? 1 : 0) - (ownOf(a) ? 1 : 0) || Object.keys(b.titles).length - Object.keys(a.titles).length || a.name.localeCompare(b.name));
  texts(); list();
  const h = location.hash.slice(1);
  show(CAT.find(x => x.q === h) || null);
}

function texts() {
  document.documentElement.lang = L;
  const own = CAT.filter(ownOf).length;
  $("sub").textContent = t("sub", { n: fmt(CAT.length), m: fmt(own) });
  $("lnk-contrib").textContent = t("contrib");
  $("lnk-contrib").href = `${REPO}/blob/main/${L === "es" ? "CONTRIBUTING.es.md" : "CONTRIBUTING.md"}`;
  $("lang").textContent = L === "es" ? "English" : "Español";
  $("q").placeholder = t("search");
  const fv = $("fam").value, wv = $("wl").value;
  $("fam").innerHTML = `<option value="">${t("allFam")}</option>` + Object.keys(ICON).map(f => `<option value="${f}">${ICON[f]} ${esc(TX[L].fam[f])}</option>`).join("");
  $("wl").innerHTML = `<option value="">${t("allLang")}</option>` + Object.entries(LANGS).map(([k, v]) => `<option value="${k}">${esc(t("wl", { l: v }))}</option>`).join("");
  $("fam").value = fv; $("wl").value = wv;
  $("own-l").textContent = t("own");
  $("empty").textContent = t("empty");
  if (cur) card(cur);
}

function list() {
  const q = norm($("q").value.trim()), f = $("fam").value, w = $("wl").value, own = $("own").checked;
  const res = CAT.filter(it => (!q || it._s.includes(q)) && (!f || it.fam === f) && (!w || it.titles[w]) && (!own || ownOf(it)));
  $("count").textContent = t("count", { n: fmt(res.length) });
  $("list").innerHTML = res.slice(0, shown).map(it => {
    const sub = [TX[L].fam[it.fam], it.sunk || it.built].filter(Boolean).join(" · ");
    return `<div class="it${cur === it ? " on" : ""}" data-q="${it.q}"><span class="ic">${ICON[it.fam] || "🚢"}</span>
      <span class="nm">${esc(it.titles[L] || it.name)}<small>${esc(sub)}</small></span>
      <span class="dp">${ownOf(it) ? '<span class="own">3D</span><br>' : ""}${fmt(-it.depth)} m</span></div>`;
  }).join("") + (res.length > shown ? `<button id="more">${t("more")}</button>` : "");
  const mb = $("more"); if (mb) mb.onclick = () => { shown += 300; list(); };
}

// ── 3D ───────────────────────────────────────────────────────────────────────
const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a2636);
scene.fog = new THREE.FogExp2(0x0a2636, 0.002);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49;
scene.add(new THREE.HemisphereLight(0xa8d8ea, 0x40372a, 1.3));
const sun = new THREE.DirectionalLight(0xfff4e0, 1.6);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
scene.add(sun, sun.target);

function seabedTex() {
  const c = document.createElement("canvas"); c.width = c.height = 256; const g = c.getContext("2d");
  g.fillStyle = "#857c69"; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(${Math.random() < 0.5 ? "60,52,40" : "170,160,140"},${Math.random() * 0.12})`; const s = 1 + Math.random() * 7; g.fillRect(Math.random() * 256, Math.random() * 256, s, s); }
  const tx = new THREE.CanvasTexture(c); tx.wrapS = tx.wrapT = THREE.RepeatWrapping; tx.colorSpace = THREE.SRGBColorSpace; return tx;
}
const bedMat = new THREE.MeshStandardMaterial({ map: seabedTex(), roughness: 1 });
const bed = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), bedMat);
bed.receiveShadow = true; scene.add(bed);
let grid = null;
const diver = new THREE.Group();
{
  const m = new THREE.MeshStandardMaterial({ color: 0xe8c33a, roughness: 0.7 });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 1.1, 4, 10), m); body.position.y = 0.75;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 12, 10), m); head.position.y = 1.62;
  const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.6, 10), new THREE.MeshStandardMaterial({ color: 0x9aa3a8 })); tank.position.set(0, 1.05, -0.22);
  diver.add(body, head, tank); diver.traverse(o => { o.castShadow = true; });
  scene.add(diver);
}
let model = null;
const gltf = new GLTFLoader(); gltf.setMeshoptDecoder(MeshoptDecoder);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect(), w = r.width, h = Math.max(1, r.height);
  renderer.setSize(w, h, false); camera.aspect = w / h;
  // On wide screens the card sits on the left: centre the model in the space to its right
  const cw = !$("card").hidden && !$("card").classList.contains("min") && w > 760 ? $("card").offsetWidth + 16 : 0;
  if (cw) camera.setViewOffset(w, h, -cw / 2, 0, w, h); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

function dispose(o) { o.traverse(x => { if (x.geometry) x.geometry.dispose(); if (x.material) [].concat(x.material).forEach(m => { for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose(); m.dispose(); }); }); }

async function build(it) {
  if (model) { scene.remove(model); dispose(model); model = null; }
  const own = ownOf(it);
  if (own && (!clean || own[1].built)) {                   // own model: the wreck, or the ship as built if there is one
    const g = await gltf.loadAsync(`models/${own[0]}/${own[1].id}/${clean ? "model_built.glb" : "model.glb"}`);
    model = g.scene;
  } else {
    model = buildWreck(THREE, it, { clean }).mesh;
  }
  model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  scene.add(model);
  // A wreck in pieces (bow, stern, debris…): the camera frames the main piece, the seabed covers them all
  const all = new THREE.Box3().setFromObject(model);
  let main = model.getObjectByName("main") || model, big = 0;
  for (const o of model.children) {                         // the biggest of the pieces («…_section»)
    if (!/_section$/.test(o.name)) continue;
    const s = new THREE.Box3().setFromObject(o).getSize(new THREE.Vector3()), v = s.x * s.z;
    if (v > big) { big = v; main = o; }
  }
  const box = new THREE.Box3().setFromObject(main), size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  const R = Math.max(size.x, size.z, size.y, 4), allS = all.getSize(new THREE.Vector3());
  const S = Math.max(200, R * 5, (Math.max(allS.x, allS.z) + Math.max(Math.abs(all.min.z - c.z), Math.abs(all.max.z - c.z), Math.abs(all.min.x - c.x), Math.abs(all.max.x - c.x))) * 1.3);
  bed.position.set(c.x, 0, c.z);
  bed.scale.set(S, 1, S); bedMat.map.repeat.set(S / 12, S / 12);
  if (grid) { scene.remove(grid); grid.geometry.dispose(); }
  const step = R > 150 ? 50 : R > 40 ? 10 : 2;
  grid = new THREE.GridHelper(Math.round(S / step) * step, Math.round(S / step), 0x2f5468, 0x2f5468);
  grid.material.transparent = true; grid.material.opacity = 0.25; grid.position.set(c.x, 0.02, c.z); scene.add(grid);
  diver.position.set(box.max.x + Math.max(2, R * 0.04), 0, c.z + size.z * 0.15);
  scene.fog.density = 0.9 / (R * 3);
  sun.position.set(c.x + R * 0.6, R * 1.5, c.z + R * 0.5); sun.target.position.copy(c);
  const sc = sun.shadow.camera; sc.left = sc.bottom = -R; sc.right = sc.top = R; sc.near = 1; sc.far = R * 4; sc.updateProjectionMatrix();
  camera.near = Math.max(0.05, R / 500); camera.far = R * 40; camera.updateProjectionMatrix();
  controls.target.set(c.x, size.y * 0.3, c.z);
  if (!build.keep) camera.position.set(c.x + R * 0.85, Math.max(R * 0.35, 3), c.z + R * 0.85);
  controls.maxDistance = R * 6; controls.minDistance = Math.max(1, R * 0.05);
  $("scale").textContent = `${fmt(size.z)} × ${fmt(size.x)} m · ${t("diver")}`;
}

function card(it) {
  const own = ownOf(it), wpL = it.titles[L] ? L : it.titles.en ? "en" : Object.keys(it.titles)[0];
  const wp = `https://${wpL}.wikipedia.org/wiki/${encodeURIComponent(it.titles[wpL].replace(/ /g, "_"))}`;
  const rows = [
    it.length && [t("length"), `${fmt(it.length)} m`], it.beam && [t("beam"), `${fmt(it.beam)} m`],
    [t("depth"), `${fmt(-(it.depth))} m${it.coast ? ` (${t("coast")})` : ""}`],
    it.built && [t("built"), it.built], it.sunk && [t("sunk"), it.sunk],
    [t("pos"), `${it.lat.toFixed(3)}, ${it.lon.toFixed(3)}`],
  ].filter(Boolean);
  const au = own && own[1].authors ? own[1].authors.map(a => esc(a.name)).join(", ") : "";
  const mtxt = own ? t(own[0] === "item" ? "ownM" : "classM", { a: au, lic: esc(own[1].license) })
    : t("gen", { v: GENERATOR_VERSION, u: `${REPO}/blob/main/${L === "es" ? "CONTRIBUTING.es.md" : "CONTRIBUTING.md"}` });
  $("card").innerHTML = `<h2><button class="tog" id="b-min" title="−/+">${$("card").classList.contains("min") ? "▸" : "▾"}</button>${ICON[it.fam] || ""} ${esc(it.titles[L] || it.name)}</h2>
    <div class="fam">${esc(TX[L].fam[it.fam])} · ${it.q}</div>
    <dl>${rows.map(([a, b]) => `<dt>${esc(a)}</dt><dd>${esc(b)}</dd>`).join("")}</dl>
    <div class="links"><a href="${wp}" target="_blank" rel="noopener">${t("wp")} (${wpL})</a>
      <a href="https://www.wikidata.org/wiki/${it.q}" target="_blank" rel="noopener">${t("wd")}</a></div>
    <div class="model${own ? " own" : ""}">${mtxt}</div>
    <div class="btns">${own && !own[1].built ? "" : `<button id="b-wreck" class="${clean ? "" : "on"}">${t("asWreck")}</button><button id="b-built" class="${clean ? "on" : ""}">${t("asBuilt")}</button>`}${own ? "" : `<button id="b-glb">${t("glb")}</button>`}</div>
    <div class="fam" style="margin-top:8px;font-size:12px">${t("approx")}</div>`;
  $("card").hidden = false; resize();
  $("b-min").onclick = () => { $("card").classList.toggle("min"); card(it); };
  const bw = $("b-wreck"), bb = $("b-built"), bg = $("b-glb");
  if (bw) bw.onclick = () => { clean = false; build.keep = true; build(it).then(() => { build.keep = false; card(it); }); };
  if (bb) bb.onclick = () => { clean = true; build.keep = true; build(it).then(() => { build.keep = false; card(it); }); };
  if (bg) bg.onclick = () => new GLTFExporter().parse(model, buf => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([buf], { type: "model/gltf-binary" }));
    a.download = `${it.q}-${clean ? "built" : "wreck"}-generic.glb`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, e => console.error(e), { binary: true });
}

async function show(it) {
  cur = it;
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", !!it && e.dataset.q === it.q));
  $("empty").hidden = !!it; $("card").hidden = !it; $("scale").textContent = "";
  if (!it) return;
  if (location.hash.slice(1) !== it.q) history.replaceState(null, "", "#" + it.q);
  clean = false;
  card(it);
  try { await build(it); } catch (e) { console.error(e); }
}

$("list").addEventListener("click", e => { const r = e.target.closest(".it"); if (r) show(CAT.find(x => x.q === r.dataset.q)); });
let qt = null;
$("q").addEventListener("input", () => { clearTimeout(qt); qt = setTimeout(() => { shown = 300; list(); }, 150); });
for (const id of ["fam", "wl", "own"]) $(id).addEventListener("change", () => { shown = 300; list(); });
$("lang").onclick = () => { L = L === "es" ? "en" : "es"; try { localStorage.setItem("s3d_lang", L); } catch { /* */ } texts(); list(); };
addEventListener("hashchange", () => { const it = CAT.find(x => x.q === location.hash.slice(1)); if (it && it !== cur) show(it); });

// ?gallery — one generic model of each family side by side (to check the generator)
async function gallery() {
  await load();
  const fams = Object.keys(ICON), W = 300, H = 190, cols = 5, out = document.createElement("canvas");
  out.width = W * cols; out.height = H * Math.ceil(fams.length / cols);
  Object.assign(out.style, { position: "fixed", inset: "0", width: "100%", height: "auto", zIndex: 9, background: "#0a2636" });
  const g = out.getContext("2d");
  renderer.setAnimationLoop(null); renderer.setSize(W, H, false); camera.aspect = W / H; camera.clearViewOffset(); camera.updateProjectionMatrix();
  $("card").hidden = true;
  for (let i = 0; i < fams.length; i++) {
    const pick = new URLSearchParams(location.search).get(fams[i]);
    const it = CAT.find(x => x.q === pick) || CAT.find(x => x.fam === fams[i] && x.length) || CAT.find(x => x.fam === fams[i]);
    if (!it) continue;
    await build(it);
    const box = new THREE.Box3().setFromObject(model), c = box.getCenter(new THREE.Vector3()), R = box.getSize(new THREE.Vector3()).length() / 2;
    camera.position.set(c.x + R * 1.25, c.y + R * 0.75, c.z + R * 1.25); camera.lookAt(c);
    renderer.render(scene, camera);
    g.drawImage(canvas, (i % cols) * W, Math.floor(i / cols) * H, W, H);
    g.fillStyle = "#fff"; g.font = "13px system-ui"; g.fillText(`${fams[i]} · ${it.name} (${Math.round(it.length || 0)} m)`, (i % cols) * W + 6, Math.floor(i / cols) * H + 16);
  }
  document.body.appendChild(out);
}

resize();
if (new URLSearchParams(location.search).has("gallery")) gallery();
else { renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); }); load(); }
window.__s3d = { scene, camera, controls, renderer, show, CAT: () => CAT, build: (it, o) => buildWreck(THREE, it, o) };
