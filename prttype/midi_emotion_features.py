"""MIDI から曲調学習用の簡易特徴（密度・ベロシティ・モード）を取る。"""

from __future__ import annotations

from pathlib import Path


def extract_midi_features(midi_path: Path) -> dict[str, float] | None:
    """notes/sec, mean_velocity_unit, major_flag(0/1), mode_known(0/1)。失敗時 None。"""
    try:
        import muspy

        music = muspy.read_midi(str(midi_path))
    except Exception:
        return None

    notes = [n for tr in music.tracks for n in tr.notes]
    if not notes:
        return None

    end_time = max((n.time + n.duration) for n in notes)
    res = float(music.resolution or 24)
    bpm = float(music.tempos[0].qpm) if music.tempos else 120.0
    seconds = max(1e-3, (end_time / res) * (60.0 / bpm))
    density = len(notes) / seconds
    vel = sum(float(n.velocity) for n in notes) / len(notes) / 127.0

    major_flag = 0.5
    mode_known = 0.0
    if music.key_signatures:
        ks = music.key_signatures[0]
        if ks.mode is not None:
            mode_known = 1.0
            is_min = str(ks.mode).lower().startswith("min") or ks.mode == 1
            major_flag = 0.0 if is_min else 1.0

    return {
        "density": float(density),
        "velocity": float(vel),
        "major_flag": float(major_flag),
        "mode_known": float(mode_known),
    }


def feature_vector(feat: dict[str, float]) -> list[float]:
    return [feat["density"], feat["velocity"], feat["major_flag"], feat["mode_known"]]
