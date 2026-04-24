from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .markdown_store import MarkdownStore
from .models import MemoryNode


def build_graph(store: MarkdownStore) -> dict[str, Any]:
    unique_nodes: dict[str, MemoryNode] = {}
    for node in store.all_nodes():
        existing = unique_nodes.get(node.id)
        if existing is None or node.updated_at > existing.updated_at:
            unique_nodes[node.id] = node
    graph_nodes = [_node_summary(node) for node in unique_nodes.values()]
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in unique_nodes.values():
        for relation in node.relations:
            key = (node.id, relation.target_id, relation.type)
            if key in seen or relation.target_id not in unique_nodes:
                continue
            seen.add(key)
            edges.append(
                {
                    "from": node.id,
                    "to": relation.target_id,
                    "type": relation.type,
                    "weight": relation.weight,
                }
            )
    return {"nodes": graph_nodes, "edges": edges}


def _node_summary(node: MemoryNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind,
        "status": node.status,
        "title": node.title,
        "salience": node.salience,
        "confidence": node.confidence,
        "pinned": node.pinned,
        "valid_from": node.valid_from,
        "valid_to": node.valid_to,
        "concept_refs": node.concept_refs,
    }


def node_detail(store: MarkdownStore, node_id: str) -> dict[str, Any] | None:
    node = store.get(node_id)
    if node is None:
        return None
    data = node.metadata()
    data["body"] = node.body
    return data


def build_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    store = MarkdownStore(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return
            if self.path == "/api/graph":
                self._send_json(HTTPStatus.OK, build_graph(store))
                return
            if self.path.startswith("/api/node/"):
                node_id = self.path[len("/api/node/") :]
                payload = node_detail(store, node_id)
                if payload is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "node not found", "id": node_id})
                    return
                self._send_json(HTTPStatus.OK, payload)
                return
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(root: Path | str = "memory", host: str = "127.0.0.1", port: int = 7070) -> None:
    handler_cls = build_handler(Path(root))
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}"
    print(f"OneMem memory inspector listening on {url}")
    print("press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.server_close()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OneMem · cognitive memory inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Martian+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
:root {
  color-scheme: dark;
  --void:        #05070a;
  --void-deep:   #020305;
  --panel:       #0a0d12;
  --panel-2:     #0e1218;
  --line:        rgba(255,255,255,0.06);
  --line-strong: rgba(255,255,255,0.14);
  --text:        #e8e6dc;
  --text-mute:   #8c93a0;
  --text-faint:  #4b5260;
  --signal:      #4ac6ff;
  --ok:          #9fe870;
  --warn:        #ff5c5c;
  --kind-episode: #4ac6ff;
  --kind-fact:    #9fe870;
  --kind-concept: #fb923c;
  --kind-summary: #d78cff;
}
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: "Martian Mono", ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px;
  color: var(--text);
  letter-spacing: 0.02em;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
  background:
    radial-gradient(ellipse 75% 55% at 12% -8%, rgba(74,198,255,0.08), transparent 65%),
    radial-gradient(ellipse 55% 45% at 100% 105%, rgba(215,140,255,0.06), transparent 65%),
    radial-gradient(circle at 1px 1px, rgba(255,255,255,0.035) 1px, transparent 0),
    linear-gradient(180deg, #05080c 0%, var(--void-deep) 100%);
  background-size: 100% 100%, 100% 100%, 28px 28px, 100% 100%;
}
.app {
  display: grid;
  grid-template-rows: auto 1fr auto;
  grid-template-columns: 1fr 420px;
  grid-template-areas: "header header" "stage detail" "footer footer";
  height: 100vh;
}
/* --- header --- */
header.topbar {
  grid-area: header;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 26px;
  border-bottom: 1px solid var(--line);
  animation: fade-down 0.8s cubic-bezier(.22,1,.36,1) both;
}
.brand {
  display: flex;
  gap: 14px;
  align-items: baseline;
  font-size: 10.5px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-mute);
}
.brand .glyph {
  color: var(--signal);
  font-size: 14px;
  line-height: 1;
  text-shadow: 0 0 12px var(--signal);
}
.brand .title {
  font-family: "Instrument Serif", "Times New Roman", serif;
  font-size: 22px;
  letter-spacing: 0.01em;
  text-transform: none;
  color: var(--text);
  font-weight: 400;
}
.brand .subtitle { color: var(--text-mute); }
.status {
  display: flex;
  gap: 22px;
  align-items: center;
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-mute);
}
.status > span { display: inline-flex; align-items: center; }
.pulse {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--ok);
  box-shadow: 0 0 10px var(--ok);
  margin-right: 10px;
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
/* --- stage --- */
.stage { grid-area: stage; position: relative; overflow: hidden; }
#graph { position: absolute; inset: 0; }
.hud {
  position: absolute;
  top: 22px; right: 22px;
  z-index: 4;
  display: flex;
  flex-direction: column;
  gap: 14px;
  pointer-events: none;
  animation: fade-down 0.9s cubic-bezier(.22,1,.36,1) 0.15s both;
}
.panel {
  position: relative;
  padding: 14px 18px;
  background: linear-gradient(180deg, rgba(14,18,24,0.82), rgba(10,13,18,0.64));
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--line);
}
.panel::before, .panel::after {
  content: "";
  position: absolute;
  width: 10px; height: 10px;
  border: 1px solid var(--signal);
  opacity: 0.9;
}
.panel::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
.panel::after  { bottom: -1px; right: -1px; border-top: none; border-left: none; }
.kpi-row { display: flex; gap: 26px; align-items: baseline; }
.kpi { display: flex; flex-direction: column; gap: 4px; }
.kpi-label {
  font-size: 9px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--text-faint);
}
.kpi-value {
  font-family: "Instrument Serif", serif;
  font-size: 32px;
  line-height: 1;
  color: var(--text);
  letter-spacing: -0.02em;
  font-feature-settings: "tnum" 1;
}
.legend {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 9px 20px;
  font-size: 9.5px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-mute);
}
.legend .item { display: flex; align-items: center; }
.legend .swatch {
  width: 7px; height: 7px;
  margin-right: 9px;
  border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
}
.legend .episode { color: var(--kind-episode); }
.legend .fact    { color: var(--kind-fact); }
.legend .concept { color: var(--kind-concept); }
.legend .summary { color: var(--kind-summary); }
/* --- reticle (empty state) --- */
.reticle {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  z-index: 2;
  opacity: 0;
  transition: opacity 0.5s ease;
}
.reticle.show { opacity: 1; }
.reticle .ring {
  width: 168px; height: 168px;
  border: 1px solid var(--line-strong);
  border-radius: 50%;
  position: relative;
  animation: rotate 22s linear infinite;
}
.reticle .ring::before,
.reticle .ring::after {
  content: "";
  position: absolute;
  inset: 18px;
  border-radius: 50%;
  border: 1px dashed var(--line-strong);
}
.reticle .ring::after {
  inset: 58px;
  border-style: solid;
  border-color: rgba(74,198,255,0.35);
  box-shadow: 0 0 24px rgba(74,198,255,0.15) inset;
}
@keyframes rotate { to { transform: rotate(360deg); } }
.reticle .caption {
  margin-top: 28px;
  font-size: 10px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--text-faint);
}
/* --- detail --- */
aside.detail {
  grid-area: detail;
  padding: 26px 28px 36px;
  overflow-y: auto;
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border-left: 1px solid var(--line);
  animation: fade-up 0.9s cubic-bezier(.22,1,.36,1) 0.3s both;
}
.detail-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 22px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
}
.detail-index {
  font-size: 10px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--text-faint);
}
.kind-pill {
  font-size: 9.5px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  padding: 5px 12px 5px 24px;
  border: 1px solid currentColor;
  color: var(--text-mute);
  position: relative;
}
.kind-pill::before {
  content: "";
  position: absolute;
  top: 50%; left: 10px;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 8px currentColor;
  transform: translateY(-50%);
}
.kind-pill[data-kind="episode"] { color: var(--kind-episode); }
.kind-pill[data-kind="fact"]    { color: var(--kind-fact); }
.kind-pill[data-kind="concept"] { color: var(--kind-concept); }
.kind-pill[data-kind="summary"] { color: var(--kind-summary); }
.detail-title {
  font-family: "Instrument Serif", serif;
  font-weight: 400;
  font-size: 28px;
  line-height: 1.22;
  letter-spacing: -0.01em;
  color: var(--text);
  margin: 0 0 26px;
}
.detail-title em { color: var(--text-mute); }
.meta-grid {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 10px 20px;
  margin-bottom: 26px;
}
.meta-grid dt {
  font-size: 9.5px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-faint);
  padding-top: 2px;
}
.meta-grid dd {
  font-size: 11px;
  color: var(--text);
  margin: 0;
  word-break: break-word;
  font-feature-settings: "tnum" 1, "zero" 1;
}
.body-label {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 9.5px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 10px;
}
.body-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--line);
}
.body-pre {
  white-space: pre-wrap;
  font-family: "Martian Mono", ui-monospace, monospace;
  font-size: 11.5px;
  line-height: 1.65;
  color: var(--text);
  background: var(--void-deep);
  border: 1px solid var(--line);
  padding: 16px 18px;
  max-height: calc(100vh - 460px);
  overflow-y: auto;
  margin: 0;
}
.body-pre::-webkit-scrollbar, aside.detail::-webkit-scrollbar { width: 8px; }
.body-pre::-webkit-scrollbar-thumb, aside.detail::-webkit-scrollbar-thumb {
  background: var(--line-strong);
  border-radius: 4px;
}
.idle-text {
  font-size: 11px;
  color: var(--text-mute);
  line-height: 1.75;
}
.idle-text code {
  background: var(--void-deep);
  border: 1px solid var(--line);
  padding: 1px 7px;
  font-size: 10.5px;
  color: var(--signal);
}
/* --- footer --- */
footer.statusbar {
  grid-area: footer;
  display: flex;
  justify-content: space-between;
  padding: 9px 26px;
  border-top: 1px solid var(--line);
  font-size: 9.5px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--text-mute);
  animation: fade-up 0.8s cubic-bezier(.22,1,.36,1) 0.4s both;
}
footer .divider {
  display: inline-block;
  width: 3px; height: 3px;
  background: var(--text-faint);
  border-radius: 50%;
  margin: 0 12px 2px;
  vertical-align: middle;
}
/* --- animations --- */
@keyframes fade-down { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fade-up   { from { opacity: 0; transform: translateY(8px); }  to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="glyph">◆</span>
      <span class="title">OneMem</span>
      <span class="subtitle">cognitive memory inspector</span>
    </div>
    <div class="status">
      <span><span class="pulse"></span>live</span>
      <span id="clock">——</span>
      <span>read-only · localhost</span>
    </div>
  </header>

  <main class="stage">
    <div id="graph"></div>
    <div class="hud">
      <div class="panel">
        <div class="kpi-row">
          <div class="kpi">
            <span class="kpi-label">nodes</span>
            <span class="kpi-value" id="k-nodes">0</span>
          </div>
          <div class="kpi">
            <span class="kpi-label">edges</span>
            <span class="kpi-value" id="k-edges">0</span>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="legend">
          <div class="item episode"><span class="swatch" style="background: var(--kind-episode);"></span>episode</div>
          <div class="item fact"><span class="swatch" style="background: var(--kind-fact);"></span>fact</div>
          <div class="item concept"><span class="swatch" style="background: var(--kind-concept);"></span>concept</div>
          <div class="item summary"><span class="swatch" style="background: var(--kind-summary);"></span>summary</div>
        </div>
      </div>
    </div>
    <div class="reticle" id="reticle">
      <div class="ring"></div>
      <div class="caption">no transmission · select a node</div>
    </div>
  </main>

  <aside class="detail">
    <div id="detail-empty">
      <div class="detail-head">
        <div class="detail-index">M-———</div>
        <div class="kind-pill">idle</div>
      </div>
      <h1 class="detail-title"><em>Awaiting</em> selection</h1>
      <p class="idle-text">
        Click any node in the graph to decode its provenance, temporal validity, confidence, and source references. Canonical bodies stream from the local Markdown store; nothing is written back from this surface.
      </p>
    </div>
    <div id="detail-loaded" style="display:none">
      <div class="detail-head">
        <div class="detail-index" id="d-index">M-001</div>
        <div class="kind-pill" id="d-kind" data-kind="fact">fact</div>
      </div>
      <h1 class="detail-title" id="d-title"></h1>
      <dl class="meta-grid" id="d-meta"></dl>
      <div class="body-label">canonical body</div>
      <pre class="body-pre" id="d-body"></pre>
    </div>
  </aside>

  <footer class="statusbar">
    <span>markdown is truth<span class="divider"></span>sidecars are views<span class="divider"></span>writes are controlled</span>
    <span id="s-right">v0.1.0 · 127.0.0.1</span>
  </footer>
</div>

<script>
const COLORS = {
  episode: { background: "rgba(74,198,255,0.20)",  border: "#4ac6ff", highlight: { background: "rgba(74,198,255,0.40)",  border: "#9deaff" } },
  fact:    { background: "rgba(159,232,112,0.18)", border: "#9fe870", highlight: { background: "rgba(159,232,112,0.36)", border: "#d5ffb0" } },
  concept: { background: "rgba(251,146,60,0.20)",  border: "#fb923c", highlight: { background: "rgba(251,146,60,0.40)",  border: "#ffc285" } },
  summary: { background: "rgba(215,140,255,0.18)", border: "#d78cff", highlight: { background: "rgba(215,140,255,0.36)", border: "#eac0ff" } },
};

const escape = (value) => String(value == null ? "" : value).replace(
  /[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c])
);

const clockEl = document.getElementById("clock");
function tick() {
  const d = new Date();
  clockEl.textContent = d.toISOString().replace("T", " ").slice(0, 19) + " utc";
}
tick();
setInterval(tick, 1000);

function tween(el, target, duration = 900) {
  const start = performance.now();
  const from = Number(el.textContent) || 0;
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (target - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function sizeFor(node) { return 10 + (Number(node.salience) || 0) * 24; }
function setReticle(show) { document.getElementById("reticle").classList.toggle("show", show); }

function renderDetail(node, indexOrder) {
  document.getElementById("detail-empty").style.display = "none";
  document.getElementById("detail-loaded").style.display = "block";
  const kind = node.kind || "fact";
  document.getElementById("d-index").textContent = "M-" + String(indexOrder).padStart(3, "0");
  const kindEl = document.getElementById("d-kind");
  kindEl.textContent = kind;
  kindEl.setAttribute("data-kind", kind);
  document.getElementById("d-title").textContent = node.title || node.id;

  const metaRows = [
    ["id",         node.id],
    ["status",     node.status],
    ["salience",   Number(node.salience ?? 0).toFixed(3)],
    ["confidence", Number(node.confidence ?? 0).toFixed(3)],
    ["pinned",     node.pinned ? "yes" : "no"],
    ["valid from", node.valid_from || "—"],
    ["valid to",   node.valid_to   || "—"],
    ["created",    node.created_at || "—"],
    ["updated",    node.updated_at || "—"],
    ["concepts",   (node.concept_refs || []).join(", ") || "—"],
    ["sources",    (node.source_refs  || []).join(", ") || "—"],
  ];
  document.getElementById("d-meta").innerHTML =
    metaRows.map(([k, v]) => `<dt>${escape(k)}</dt><dd>${escape(v)}</dd>`).join("");
  document.getElementById("d-body").textContent = node.body || "";
}

async function loadGraph() {
  if (typeof vis === "undefined" || !vis.Network || !vis.DataSet) {
    document.getElementById("detail-empty").innerHTML =
      '<h1 class="detail-title" style="color:var(--warn)">vis-network failed to load</h1>' +
      '<p class="idle-text">The graph library could not be fetched from unpkg.com. Check your connection or proxy.</p>';
    return;
  }
  const response = await fetch("/api/graph");
  if (!response.ok) throw new Error("graph endpoint returned HTTP " + response.status);
  const data = await response.json();

  tween(document.getElementById("k-nodes"), data.nodes.length);
  tween(document.getElementById("k-edges"), data.edges.length);

  if (!data.nodes.length) {
    setReticle(true);
    document.getElementById("detail-empty").innerHTML =
      '<div class="detail-head"><div class="detail-index">M-———</div><div class="kind-pill">empty</div></div>' +
      '<h1 class="detail-title"><em>No memory</em> yet</h1>' +
      '<p class="idle-text">Capture an observation and run <code>onemem flush</code> to see nodes materialize here.</p>';
    return;
  }

  setReticle(true);

  const indexById = new Map(
    [...data.nodes].map((n) => n.id).sort().map((id, i) => [id, i + 1])
  );

  const visNodes = new vis.DataSet(data.nodes.map((n) => {
    const title = (n.title || n.id);
    return {
      id: n.id,
      label: title.length > 36 ? title.slice(0, 33) + "…" : title,
      title: `${n.kind} · ${n.status} · salience=${Number(n.salience).toFixed(2)}`,
      color: COLORS[n.kind] || COLORS.fact,
      size: sizeFor(n),
    };
  }));
  const visEdges = new vis.DataSet(data.edges.map((e) => ({
    from: e.from,
    to: e.to,
    label: e.type,
    arrows: { to: { enabled: true, scaleFactor: 0.4 } },
    color: { color: "rgba(255,255,255,0.18)", highlight: "#4ac6ff" },
    font: { color: "#6b7280", size: 9, strokeWidth: 0, align: "middle", face: "Martian Mono" },
    smooth: { type: "continuous", roundness: 0.2 },
    width: 0.5 + 1.6 * (Number(e.weight) || 0),
  })));

  const network = new vis.Network(
    document.getElementById("graph"),
    { nodes: visNodes, edges: visEdges },
    {
      nodes: {
        shape: "dot",
        borderWidth: 1.4,
        font: { color: "#e8e6dc", size: 11, face: "Martian Mono", strokeWidth: 0, vadjust: -14 },
        shadow: { enabled: true, color: "rgba(0,0,0,0.55)", size: 14, x: 0, y: 2 },
      },
      edges: { hoverWidth: 1.2 },
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: { gravitationalConstant: -46, springLength: 150, springConstant: 0.04, damping: 0.55 },
        stabilization: { iterations: 220, updateInterval: 25 },
      },
      interaction: { hover: true, tooltipDelay: 160, zoomView: true, dragView: true, navigationButtons: false },
    }
  );

  network.on("stabilizationIterationsDone", () => {
    network.setOptions({ physics: { enabled: false } });
  });

  network.on("click", async (params) => {
    if (!params.nodes.length) {
      setReticle(true);
      document.getElementById("detail-empty").style.display = "block";
      document.getElementById("detail-loaded").style.display = "none";
      return;
    }
    setReticle(false);
    const id = params.nodes[0];
    try {
      const r = await fetch("/api/node/" + encodeURIComponent(id));
      if (!r.ok) throw new Error();
      const node = await r.json();
      renderDetail(node, indexById.get(id) || 0);
    } catch {
      setReticle(true);
    }
  });
}

loadGraph().catch((error) => {
  document.getElementById("detail-empty").innerHTML =
    '<h1 class="detail-title" style="color:var(--warn)">Load failed</h1>' +
    '<p class="idle-text">' + escape(error) + '</p>';
});
</script>
</body>
</html>
"""
