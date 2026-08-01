"""prior_pairs ゲート画面用のローカルサーバ。

起動:
  cd prttype
  ..\\venv\\Scripts\\python.exe scripts\\prior_pair_gate_server.py

ブラウザで http://127.0.0.1:8765 を開く。
  A = accept / R = reject
  1-5 = score_fit
変更は candidates.jsonl に即保存される。
"""

from __future__ import annotations

import json
import mimetypes
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

PAIRS_ROOT = SCRIPT_DIR / "data" / "prior_pairs"
MANIFEST = PAIRS_ROOT / "manifests" / "candidates.jsonl"
MIDI_DIR = PAIRS_ROOT / "midi"
WAV_DIR = PAIRS_ROOT / "wav"
HOST = "127.0.0.1"
PORT = 8765
SR = 22050


def midi_to_preview_wav(midi_path: Path, wav_path: Path) -> None:
    """ブラウザ再生用の簡易プレビュー WAV（サイン波合成。音色は仮）。"""
    import muspy

    music = muspy.read_midi(midi_path)
    qpm = float(music.tempos[0].qpm) if music.tempos else 120.0
    # muspy resolution: ticks per quarter
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
        buf[i0 : i0 + length] += 0.15 * env * np.sin(2.0 * np.pi * f * t)

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


def ensure_preview_wav(pair_id: str) -> Path:
    midi_path = MIDI_DIR / f"{pair_id}.mid"
    wav_path = WAV_DIR / f"{pair_id}.wav"
    if not midi_path.is_file():
        raise FileNotFoundError(midi_path)
    if (
        wav_path.is_file()
        and wav_path.stat().st_mtime >= midi_path.stat().st_mtime
    ):
        return wav_path
    midi_to_preview_wav(midi_path, wav_path)
    return wav_path


HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<title>prior_pairs gate</title>
<style>
  :root {
    --bg: #141414;
    --panel: #1e1e1e;
    --text: #f2f2f2;
    --muted: #9a9a9a;
    --line: #333;
    --accept: #2f6f4e;
    --reject: #7a3030;
    --pending: #3a3a3a;
    --accent: #c4a574;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", "Hiragino Sans", sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    position: sticky; top: 0; z-index: 5;
    display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
    padding: 0.75rem 1rem;
    background: #101010;
    border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 1rem; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
  .hint { color: var(--muted); font-size: 0.85rem; }
  .stats { margin-left: auto; color: var(--accent); font-size: 0.9rem; }
  table { width: 100%; border-collapse: collapse; }
  th, td {
    padding: 0.55rem 0.7rem;
    border-bottom: 1px solid var(--line);
    vertical-align: middle;
    text-align: left;
  }
  th {
    position: sticky; top: 96px; z-index: 4;
    background: #181818;
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  tr:hover td { background: #222; }
  tr.active td { outline: 1px solid var(--accent); background: #242018; }
  .prompt { font-size: 0.95rem; max-width: 28rem; }
  .meta { color: var(--muted); font-size: 0.8rem; white-space: nowrap; }
  .status {
    display: inline-block;
    min-width: 4.5rem;
    text-align: center;
    padding: 0.2rem 0.45rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .status.pending { background: var(--pending); }
  .status.accept { background: var(--accept); }
  .status.reject { background: var(--reject); }
  .score {
    display: inline-block;
    min-width: 1.6rem;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }
  .btns { display: flex; gap: 0.25rem; flex-wrap: wrap; }
  button {
    border: 1px solid #555;
    background: #2a2a2a;
    color: var(--text);
    border-radius: 4px;
    padding: 0.25rem 0.45rem;
    cursor: pointer;
    font-size: 0.8rem;
  }
  button:hover { border-color: var(--accent); }
  button.A { border-color: #3d8f62; }
  button.R { border-color: #a05050; }
  button.scorebtn { min-width: 1.7rem; }
  button.play { color: var(--accent); }
  .filter select {
    background: #222; color: var(--text); border: 1px solid #555;
    border-radius: 4px; padding: 0.2rem 0.4rem;
  }
  #playerBar {
    display: flex; align-items: center; gap: 0.75rem;
    width: 100%;
    padding-top: 0.35rem;
  }
  #audioPlayer { width: min(420px, 100%); height: 32px; }
</style>
</head>
<body>
<header>
  <h1>prior_pairs gate</h1>
  <span class="hint">選択行で <b>A</b>=accept / <b>R</b>=reject / <b>1–5</b>=score_fit / <b>Space</b>=再生（WAVプレビュー）</span>
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
    <span class="hint" id="nowPlaying">未選択（初回再生は変換で数秒かかることがあります）</span>
  </div>
</header>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>prompt</th>
      <th>target / energy / bpm</th>
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
  return rows.map((r, i) => ({r, i})).filter(({r}) => f === 'all' || r.gate.status === f).map(({i}) => i);
}

function updateStats() {
  const c = {pending:0, accept:0, reject:0};
  for (const r of rows) c[r.gate.status] = (c[r.gate.status]||0)+1;
  stats.textContent = `pending ${c.pending||0} · accept ${c.accept||0} · reject ${c.reject||0} · total ${rows.length}`;
}

function statusClass(s) { return 'status ' + (s || 'pending'); }

function render() {
  const vis = new Set(visibleIndices());
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    if (!vis.has(i)) return;
    const tr = document.createElement('tr');
    if (i === active) tr.classList.add('active');
    tr.dataset.idx = i;
    const st = r.gate.status || 'pending';
    const sc = r.gate.score_fit;
    tr.innerHTML = `
      <td class="meta">${i+1}<br><span>${r.id}</span></td>
      <td class="prompt">${escapeHtml(r.prompt)}</td>
      <td class="meta">${r.emotion_target} · ${r.structure.energy} · BPM ${r.structure.bpm}<br>${r.structure.progression} / ${r.structure.key}</td>
      <td><span class="${statusClass(st)}" id="st-${i}">${st}</span></td>
      <td><span class="score" id="sc-${i}">${sc == null ? '—' : sc}</span></td>
      <td class="btns">
        <button class="play" data-act="play" data-i="${i}">▶</button>
        <button data-act="open" data-i="${i}">OS</button>
        <button class="A" data-act="A" data-i="${i}">A</button>
        <button class="R" data-act="R" data-i="${i}">R</button>
        ${[1,2,3,4,5].map(n => `<button class="scorebtn" data-act="score" data-n="${n}" data-i="${i}">${n}</button>`).join('')}
      </td>`;
    tr.addEventListener('click', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      setActive(i);
    });
    tbody.appendChild(tr);
  });
  updateStats();
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function patch(i, body) {
  const res = await fetch('/api/gate/' + encodeURIComponent(rows[i].id), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!res.ok) { alert('save failed'); return; }
  const updated = await res.json();
  rows[i].gate = updated.gate;
  const st = document.getElementById('st-' + i);
  const sc = document.getElementById('sc-' + i);
  if (st) { st.textContent = updated.gate.status; st.className = statusClass(updated.gate.status); }
  if (sc) sc.textContent = updated.gate.score_fit == null ? '—' : updated.gate.score_fit;
  updateStats();
}

function setActive(i) {
  active = i;
  document.querySelectorAll('tr.active').forEach(el => el.classList.remove('active'));
  const tr = tbody.querySelector(`tr[data-idx="${i}"]`);
  if (tr) {
    tr.classList.add('active');
    tr.scrollIntoView({block: 'nearest'});
  }
}

async function play(i) {
  setActive(i);
  nowPlaying.textContent = '変換中… ' + rows[i].id;
  const res = await fetch('/api/preview/' + encodeURIComponent(rows[i].id));
  if (!res.ok) {
    nowPlaying.textContent = 'プレビュー失敗';
    alert('preview failed');
    return;
  }
  const data = await res.json();
  audioPlayer.src = data.url + '?t=' + Date.now();
  nowPlaying.textContent = rows[i].id + ' · ' + rows[i].prompt + '（サイン波プレビュー）';
  try {
    await audioPlayer.play();
  } catch (e) {
    nowPlaying.textContent = '再生ブロック: 画面の再生ボタンを押してください';
  }
}

tbody.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  const i = Number(btn.dataset.i);
  const act = btn.dataset.act;
  if (act === 'play') play(i);
  if (act === 'open') fetch('/api/open/' + encodeURIComponent(rows[i].id));
  if (act === 'A') patch(i, {status: 'accept'});
  if (act === 'R') patch(i, {status: 'reject'});
  if (act === 'score') patch(i, {score_fit: Number(btn.dataset.n)});
});

filterEl.addEventListener('change', () => {
  render();
  const vis = visibleIndices();
  if (vis.length && !vis.includes(active)) setActive(vis[0]);
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  const vis = visibleIndices();
  if (!vis.length) return;
  let pos = vis.indexOf(active);
  if (pos < 0) pos = 0;
  if (e.key === 'ArrowDown' || e.key === 'j') {
    e.preventDefault();
    setActive(vis[Math.min(pos + 1, vis.length - 1)]);
  } else if (e.key === 'ArrowUp' || e.key === 'k') {
    e.preventDefault();
    setActive(vis[Math.max(pos - 1, 0)]);
  } else if (e.key === 'a' || e.key === 'A') {
    e.preventDefault();
    patch(active, {status: 'accept'});
  } else if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    patch(active, {status: 'reject'});
  } else if (e.key >= '1' && e.key <= '5') {
    e.preventDefault();
    patch(active, {score_fit: Number(e.key)});
  } else if (e.key === ' ') {
    e.preventDefault();
    play(active);
  }
});

fetch('/api/candidates').then(r => r.json()).then(data => {
  rows = data;
  render();
  const vis = visibleIndices();
  if (vis.length) setActive(vis[0]);
});
</script>
</body>
</html>
"""


def load_rows() -> list[dict]:
    rows: list[dict] = []
    if not MANIFEST.is_file():
        return rows
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def save_rows(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/candidates":
            body = json.dumps(load_rows(), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path.startswith("/midi/"):
            name = path[len("/midi/") :]
            midi_path = MIDI_DIR / name
            if not midi_path.is_file() or midi_path.suffix.lower() != ".mid":
                self._send(404, b"not found", "text/plain")
                return
            data = midi_path.read_bytes()
            ctype = mimetypes.guess_type(str(midi_path))[0] or "audio/midi"
            self._send(200, data, ctype)
            return
        if path.startswith("/wav/"):
            name = path[len("/wav/") :]
            wav_path = WAV_DIR / name
            if not wav_path.is_file() or wav_path.suffix.lower() != ".wav":
                self._send(404, b"not found", "text/plain")
                return
            data = wav_path.read_bytes()
            self._send(200, data, "audio/wav")
            return
        if path.startswith("/api/preview/"):
            pair_id = path[len("/api/preview/") :]
            try:
                wav_path = ensure_preview_wav(pair_id)
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
            midi_path = MIDI_DIR / f"{pair_id}.mid"
            if not midi_path.is_file():
                self._send(404, b"not found", "text/plain")
                return
            os.startfile(str(midi_path))  # noqa: S606 — local gate helper
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
            if row.get("id") == pair_id:
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
                found = row
                break
        if found is None:
            self._send(404, b"id not found", "text/plain")
            return
        save_rows(rows)
        body = json.dumps(found, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"manifest がありません: {MANIFEST}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"prior_pairs gate: http://{HOST}:{PORT}")
    print(f"manifest: {MANIFEST}")
    print("Ctrl+C で終了")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
