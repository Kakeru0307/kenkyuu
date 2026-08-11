"""form_manifest ゲート画面用のローカルサーバ。

起動:
  cd prttype
  .\\.venv\\Scripts\\python.exe scripts\\form_gate_server.py

ブラウザで http://127.0.0.1:8766 を開く。
  A = accept / R = reject / 1-5 = score_fit / Space = 再生
変更は midi/form_*.json の gate に即保存。

集約:
  python scripts/form_gate_server.py --export
  → data/form_candidates/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MIDI_DIR = SCRIPT_DIR / "midi"
WAV_DIR = SCRIPT_DIR / "data" / "form_candidates" / "wav"
CANDIDATES_DIR = SCRIPT_DIR / "data" / "form_candidates"
CANDIDATES_PATH = CANDIDATES_DIR / "candidates.jsonl"
HOST = "127.0.0.1"
PORT = 8766
SR = 22050


def list_manifests() -> list[Path]:
    if not MIDI_DIR.is_dir():
        return []
    return sorted(MIDI_DIR.glob("form_*.json"))


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in list_manifests():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row.setdefault("id", path.stem)
        row.setdefault("gate", {"status": "pending", "score_fit": None, "note": ""})
        row["_path"] = str(path)
        rows.append(row)
    return rows


def save_row(row: dict) -> None:
    path = Path(row["_path"])
    payload = {k: v for k, v in row.items() if not k.startswith("_")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_candidates() -> Path:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    lines = []
    for row in rows:
        payload = {k: v for k, v in row.items() if not k.startswith("_")}
        lines.append(json.dumps(payload, ensure_ascii=False))
    CANDIDATES_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return CANDIDATES_PATH


def midi_to_preview_wav(midi_path: Path, wav_path: Path) -> None:
    import muspy

    music = muspy.read_midi(midi_path)
    qpm = float(music.tempos[0].qpm) if music.tempos else 120.0
    tpb = int(music.resolution or 24)
    sec_per_tick = (60.0 / qpm) / max(tpb, 1)

    notes: list[tuple[float, float, int]] = []
    max_t = 0.0
    for track in music.tracks:
        if track.is_drum:
            continue
        for n in track.notes:
            start = float(n.time) * sec_per_tick
            dur = max(0.05, float(n.duration) * sec_per_tick)
            notes.append((start, dur, int(n.pitch)))
            max_t = max(max_t, start + dur)
    if not notes:
        max_t = 1.0

    n_samples = int(max_t * SR) + SR // 2
    buf = np.zeros(n_samples, dtype=np.float32)
    for start, dur, pitch in notes:
        f = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        i0 = int(start * SR)
        length = int(dur * SR)
        if i0 >= n_samples:
            continue
        length = min(length, n_samples - i0)
        t = np.arange(length, dtype=np.float32) / SR
        env = np.linspace(1.0, 0.15, length, dtype=np.float32)
        buf[i0 : i0 + length] += 0.12 * env * np.sin(2.0 * np.pi * f * t)

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1e-6:
        buf = buf / peak * 0.85
    pcm = (buf * 32767.0).astype(np.int16)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def ensure_preview_wav(row: dict) -> Path:
    midi_name = row.get("midi") or f"{row['id']}.mid"
    midi_path = MIDI_DIR / midi_name
    wav_path = WAV_DIR / f"{row['id']}.wav"
    if not midi_path.is_file():
        raise FileNotFoundError(midi_path)
    if wav_path.is_file() and wav_path.stat().st_mtime >= midi_path.stat().st_mtime:
        return wav_path
    midi_to_preview_wav(midi_path, wav_path)
    return wav_path


HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>form_candidates gate</title>
<style>
  :root {
    --bg: #141414; --panel: #1e1e1e; --text: #f2f2f2; --muted: #9a9a9a;
    --line: #333; --accept: #2f6f4e; --reject: #7a3030; --pending: #3a3a3a;
    --accent: #c4a574;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", "Hiragino Sans", sans-serif; background: var(--bg); color: var(--text); }
  header {
    position: sticky; top: 0; z-index: 5;
    display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
    padding: 0.75rem 1rem; background: #101010; border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 1rem; margin: 0; font-weight: 600; }
  .hint { color: var(--muted); font-size: 0.85rem; }
  .stats { margin-left: auto; color: var(--accent); font-size: 0.9rem; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
  th { position: sticky; top: 96px; z-index: 4; background: #181818; color: var(--muted); font-size: 0.75rem; }
  tr:hover td { background: #222; }
  tr.active td { outline: 1px solid var(--accent); background: #242018; }
  .prompt { font-size: 0.95rem; max-width: 22rem; }
  .meta { color: var(--muted); font-size: 0.8rem; }
  .status { display: inline-block; min-width: 4.5rem; text-align: center; padding: 0.2rem 0.45rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
  .status.pending { background: var(--pending); }
  .status.accept { background: var(--accept); }
  .status.reject { background: var(--reject); }
  .btns { display: flex; gap: 0.25rem; flex-wrap: wrap; }
  button { border: 1px solid #555; background: #2a2a2a; color: var(--text); border-radius: 4px; padding: 0.25rem 0.45rem; cursor: pointer; font-size: 0.8rem; }
  button:hover { border-color: var(--accent); }
  button.A { border-color: #3d8f62; }
  button.R { border-color: #a05050; }
  button.play { color: var(--accent); }
  .filter select { background: #222; color: var(--text); border: 1px solid #555; border-radius: 4px; padding: 0.2rem 0.4rem; }
  #playerBar { display: flex; align-items: center; gap: 0.75rem; width: 100%; padding-top: 0.35rem; }
  #audioPlayer { width: min(420px, 100%); height: 32px; }
</style>
</head>
<body>
<header>
  <h1>form_candidates gate</h1>
  <span class="hint">選択行で <b>A</b>=accept / <b>R</b>=reject / <b>1–5</b>=score_fit / <b>Space</b>=再生</span>
  <label class="filter">filter
    <select id="filter">
      <option value="all">all</option>
      <option value="pending">pending</option>
      <option value="accept">accept</option>
      <option value="reject">reject</option>
    </select>
  </label>
  <span class="stats" id="stats"></span>
  <div id="playerBar">
    <audio id="audioPlayer" controls preload="none"></audio>
    <span class="hint" id="nowPlaying">未選択</span>
  </div>
</header>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>text / home</th>
      <th>sections</th>
      <th>status</th>
      <th>score</th>
      <th>actions</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<script>
let rows = [];
let active = 0;
const tbody = document.getElementById('tbody');
const stats = document.getElementById('stats');
const filterEl = document.getElementById('filter');
const audioPlayer = document.getElementById('audioPlayer');
const nowPlaying = document.getElementById('nowPlaying');

function visibleIndices() {
  const f = filterEl.value;
  return rows.map((r, i) => ({r, i})).filter(({r}) => f === 'all' || (r.gate?.status || 'pending') === f).map(({i}) => i);
}
function updateStats() {
  const c = {pending:0, accept:0, reject:0};
  for (const r of rows) c[r.gate?.status || 'pending'] = (c[r.gate?.status || 'pending']||0)+1;
  stats.textContent = `pending ${c.pending||0} · accept ${c.accept||0} · reject ${c.reject||0} · total ${rows.length}`;
}
function statusClass(s) { return 'status ' + (s || 'pending'); }
function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function sectionSummary(r) {
  return (r.sections || []).map(s => `${s.label}:${s.progression}`).join(' → ');
}
function render() {
  const vis = new Set(visibleIndices());
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    if (!vis.has(i)) return;
    const tr = document.createElement('tr');
    if (i === active) tr.classList.add('active');
    const st = r.gate?.status || 'pending';
    const sc = r.gate?.score_fit;
    const va = r.va || [0, 0];
    tr.innerHTML = `
      <td class="meta">${i+1}<br><span>${escapeHtml(r.id)}</span></td>
      <td class="prompt">${escapeHtml(r.text || '(no text)')}<br>
        <span class="meta">home=${escapeHtml(r.home_progression)} / ${escapeHtml(r.key)} / BPM ${Number(r.bpm).toFixed(1)}
        · V=${Number(va[0]).toFixed(2)} A=${Number(va[1]).toFixed(2)}</span></td>
      <td class="meta">${escapeHtml(sectionSummary(r))}</td>
      <td><span class="${statusClass(st)}" id="st-${i}">${st}</span></td>
      <td><span class="score" id="sc-${i}">${sc == null ? '—' : sc}</span></td>
      <td class="btns">
        <button class="play" data-act="play" data-i="${i}">▶</button>
        <button data-act="open" data-i="${i}">OS</button>
        <button class="A" data-act="A" data-i="${i}">A</button>
        <button class="R" data-act="R" data-i="${i}">R</button>
        ${[1,2,3,4,5].map(n => `<button data-act="score" data-n="${n}" data-i="${i}">${n}</button>`).join('')}
      </td>`;
    tr.addEventListener('click', (e) => { if (e.target.tagName !== 'BUTTON') setActive(i); });
    tbody.appendChild(tr);
  });
  updateStats();
}
async function patch(i, body) {
  const res = await fetch('/api/gate/' + encodeURIComponent(rows[i].id), {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  if (!res.ok) { alert('patch failed'); return; }
  const updated = await res.json();
  rows[i].gate = updated.gate;
  const st = document.getElementById('st-' + i);
  const sc = document.getElementById('sc-' + i);
  if (st) { st.textContent = updated.gate.status; st.className = statusClass(updated.gate.status); }
  if (sc) sc.textContent = updated.gate.score_fit == null ? '—' : updated.gate.score_fit;
  updateStats();
}
function setActive(i) { active = i; render(); }
async function play(i) {
  setActive(i);
  nowPlaying.textContent = '変換中… ' + rows[i].id;
  const res = await fetch('/api/preview/' + encodeURIComponent(rows[i].id));
  if (!res.ok) { nowPlaying.textContent = 'preview failed'; return; }
  const data = await res.json();
  audioPlayer.src = data.url + '?t=' + Date.now();
  nowPlaying.textContent = rows[i].id;
  await audioPlayer.play().catch(() => {});
}
async function openMidi(i) {
  await fetch('/api/open/' + encodeURIComponent(rows[i].id));
}
tbody.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  const i = Number(btn.dataset.i);
  const act = btn.dataset.act;
  if (act === 'play') play(i);
  else if (act === 'open') openMidi(i);
  else if (act === 'A') patch(i, {status: 'accept'});
  else if (act === 'R') patch(i, {status: 'reject'});
  else if (act === 'score') patch(i, {score_fit: Number(btn.dataset.n)});
});
filterEl.addEventListener('change', render);
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const vis = visibleIndices();
  if (!vis.length) return;
  let idx = vis.indexOf(active);
  if (idx < 0) idx = 0;
  if (e.key === 'ArrowDown') { e.preventDefault(); setActive(vis[Math.min(idx + 1, vis.length - 1)]); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(vis[Math.max(idx - 1, 0)]); }
  else if (e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); play(active); }
  else if (e.key === 'a' || e.key === 'A') patch(active, {status: 'accept'});
  else if (e.key === 'r' || e.key === 'R') patch(active, {status: 'reject'});
  else if (['1','2','3','4','5'].includes(e.key)) patch(active, {score_fit: Number(e.key)});
});
fetch('/api/rows').then(r => r.json()).then(data => { rows = data; if (rows.length) active = 0; render(); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/rows":
            rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in load_rows()]
            body = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path.startswith("/wav/"):
            wav_path = WAV_DIR / path[len("/wav/") :]
            if not wav_path.is_file() or wav_path.suffix.lower() != ".wav":
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, wav_path.read_bytes(), "audio/wav")
            return
        if path.startswith("/api/preview/"):
            pair_id = path[len("/api/preview/") :]
            rows = load_rows()
            row = next((r for r in rows if r.get("id") == pair_id), None)
            if row is None:
                self._send(404, b"id not found", "text/plain")
                return
            try:
                wav_path = ensure_preview_wav(row)
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
                return
            body = json.dumps(
                {"url": f"/wav/{wav_path.name}", "id": pair_id},
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path.startswith("/api/open/"):
            pair_id = path[len("/api/open/") :]
            rows = load_rows()
            row = next((r for r in rows if r.get("id") == pair_id), None)
            if row is None:
                self._send(404, b"id not found", "text/plain")
                return
            midi_name = row.get("midi") or f"{pair_id}.mid"
            midi_path = MIDI_DIR / midi_name
            if not midi_path.is_file():
                self._send(404, b"midi not found", "text/plain")
                return
            os.startfile(str(midi_path))  # noqa: S606
            self._send(200, b'{"ok":true}', "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/gate/"):
            self._send(404, b"not found", "text/plain")
            return
        pair_id = path[len("/api/gate/") :]
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        rows = load_rows()
        found = None
        for row in rows:
            if row.get("id") != pair_id:
                continue
            gate = dict(row.get("gate") or {})
            if "status" in payload:
                status = str(payload["status"])
                if status not in ("pending", "accept", "reject"):
                    self._send(400, b"bad status", "text/plain")
                    return
                gate["status"] = status
            if "score_fit" in payload:
                score = payload["score_fit"]
                if score is not None:
                    score = int(score)
                    if score < 1 or score > 5:
                        self._send(400, b"bad score", "text/plain")
                        return
                gate["score_fit"] = score
            if "note" in payload:
                gate["note"] = str(payload["note"] or "")
            row["gate"] = gate
            save_row(row)
            found = {k: v for k, v in row.items() if not k.startswith("_")}
            break
        if found is None:
            self._send(404, b"id not found", "text/plain")
            return
        body = json.dumps(found, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="form_manifest ゲート UI")
    parser.add_argument(
        "--export",
        action="store_true",
        help="midi/form_*.json を data/form_candidates/candidates.jsonl に集約して終了",
    )
    args = parser.parse_args()
    if args.export:
        path = export_candidates()
        n = len(load_rows())
        print(f"exported {n} rows → {path}")
        return

    n = len(list_manifests())
    if n == 0:
        print(f"警告: form_*.json がありません: {MIDI_DIR}")
        print("先に generate_form.py で生成してください。")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"form gate: http://{HOST}:{PORT}")
    print(f"manifests: {MIDI_DIR}/form_*.json ({n} files)")
    print("Ctrl+C で終了 / --export で candidates.jsonl 集約")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
