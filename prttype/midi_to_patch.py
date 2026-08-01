"""MIDI を ViTex 互換の 128x128 パッチ列に変換する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import muspy
import numpy as np

TICKS_PER_BAR = 16
PATCH_BARS = 8
PATCH_TICKS = PATCH_BARS * TICKS_PER_BAR
NUM_CATEGORIES = 11


@dataclass
class MidiPatch:
    tonal: np.ndarray
    drum: np.ndarray
    bar_index: int

    @property
    def tonal_chw(self) -> np.ndarray:
        """(11, 128, 128) = C, time, pitch"""
        return self.tonal.transpose(2, 0, 1)

    @property
    def drum_chw(self) -> np.ndarray:
        """(1, 128, 128) = C, time, pitch"""
        return self.drum[np.newaxis, ...]


def _category(program: int) -> int:
    category = program // 8
    return min(category, NUM_CATEGORIES - 1)


def _collect_tonal_notes(music: muspy.Music) -> np.ndarray:
    notes_list: list[list[int]] = []
    for track in music.tracks:
        if track.is_drum:
            continue
        category = _category(track.program)
        for note in track:
            duration = note.duration + int(note.duration == 0)
            notes_list.append([note.time, note.pitch, category, duration])

    if not notes_list:
        return np.zeros((0, 4), dtype=np.int32)

    notes = np.array(notes_list, dtype=np.int32)
    return notes[notes[:, 0].argsort()]


def _collect_drum_notes(music: muspy.Music) -> np.ndarray:
    notes_list: list[list[int]] = []
    for track in music.tracks:
        if not track.is_drum:
            continue
        for note in track:
            duration = note.duration + int(note.duration == 0)
            notes_list.append([note.time, note.pitch, duration])

    if not notes_list:
        return np.zeros((0, 3), dtype=np.int32)

    notes = np.array(notes_list, dtype=np.int32)
    return notes[notes[:, 0].argsort()]


def _render_tonal_patch(tonal_notes: np.ndarray, bar_index: int) -> np.ndarray:
    """(128, 128, 11) = time, pitch, category"""
    pianoroll = np.zeros((PATCH_TICKS, 128, NUM_CATEGORIES), dtype=np.int32)
    start_tick = bar_index * TICKS_PER_BAR

    for onset, pitch, category, duration in tonal_notes:
        w = int(onset - start_tick)
        if w < 0 or w >= PATCH_TICKS:
            continue
        dw = int(duration - 1)
        end_w = min(w + dw, PATCH_TICKS)
        pianoroll[w, pitch, category] = 1
        if end_w > w + 1:
            pianoroll[w + 1 : end_w, pitch, category] = 2

    return pianoroll


def _render_drum_patch(drum_notes: np.ndarray, bar_index: int) -> np.ndarray:
    """(128, 128) = time, pitch"""
    pianoroll = np.zeros((PATCH_TICKS, 128), dtype=np.int32)
    start_tick = bar_index * TICKS_PER_BAR

    for onset, pitch, duration in drum_notes:
        w = int(onset - start_tick)
        if w < 0 or w >= PATCH_TICKS:
            continue
        dw = int(duration - 1)
        end_w = min(w + dw, PATCH_TICKS)
        pianoroll[w, pitch] = 1
        if end_w > w + 1:
            pianoroll[w + 1 : end_w, pitch] = 2

    return pianoroll


def midi_to_patches(midi_path: str | Path) -> list[MidiPatch]:
    music = muspy.read_midi(midi_path).adjust_resolution(4)
    tonal_notes = _collect_tonal_notes(music)
    drum_notes = _collect_drum_notes(music)

    end_time = music.get_end_time()
    bar_num = (end_time // TICKS_PER_BAR) + 1
    num_patches = max(0, bar_num - PATCH_BARS + 1)

    patches: list[MidiPatch] = []
    for bar_index in range(num_patches):
        tonal = _render_tonal_patch(tonal_notes, bar_index)
        drum = _render_drum_patch(drum_notes, bar_index)
        patches.append(MidiPatch(tonal=tonal, drum=drum, bar_index=bar_index))

    return patches


def normalize_pianoroll(pianoroll: np.ndarray) -> np.ndarray:
    """0/1/2 を 0.0/0.5/1.0 に変換。"""
    return pianoroll.astype(np.float32) / 2.0


def denormalize_pianoroll(values: np.ndarray) -> np.ndarray:
    """連続値を 0/1/2 に戻す。"""
    quantized = np.zeros_like(values, dtype=np.int32)
    quantized[values >= 0.25] = 1
    quantized[values >= 0.75] = 2
    return quantized


def denormalize_pianoroll_inference(
    values: np.ndarray,
    *,
    onset_th: float = 0.5,
    sustain_th: float = 0.85,
) -> np.ndarray:
    """推論用: しきい値で 0/1/2 に量子化する。

    onset_th を下げると弱い発音も拾う（単音リードなど値が低い出力向け）。
    """
    quantized = np.zeros_like(values, dtype=np.int32)
    quantized[values >= onset_th] = 1
    quantized[values >= sustain_th] = 2
    return quantized


def find_temporal_peaks(
    values: np.ndarray,
    *,
    relative_threshold: float = 0.2,
    min_distance: int = 2,
    relative_prominence: float = 0.01,
    rise_relative: float = 0.04,
) -> list[int]:
    """1次元 envelope から局所ピーク／立ち上がりを検出する。

    固定グリッドへスナップせず、モデル出力の山と立ち上がりを onset 候補にする。
    U-Net 出力は平坦寄りになりやすいので、局所最大に加えて正の差分ピークも採る。
    それでも候補が無ければ空リストを返し、呼び出し側で従来デコードへフォールバックする。
    """
    signal = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape={signal.shape}")
    if signal.size == 0:
        return []

    peak_max = float(signal.max())
    if peak_max <= 0.0:
        return []

    threshold = peak_max * float(relative_threshold)
    prominence_th = peak_max * float(relative_prominence)
    rise_th = peak_max * float(rise_relative)
    radius = max(1, int(min_distance))
    candidates: list[int] = []

    for tick, value in enumerate(signal):
        value_f = float(value)
        if value_f < threshold:
            continue
        left = float(signal[tick - 1]) if tick > 0 else float("-inf")
        right = (
            float(signal[tick + 1])
            if tick + 1 < signal.size
            else float("-inf")
        )
        is_local_max = value_f >= left and value_f >= right and (
            value_f > left or value_f > right
        )
        rise = value_f - left if np.isfinite(left) else value_f
        is_rise = rise >= rise_th and value_f >= right

        if not (is_local_max or is_rise):
            continue

        left_min = (
            float(signal[max(0, tick - radius) : tick + 1].min())
            if tick > 0
            else 0.0
        )
        right_min = (
            float(signal[tick : min(signal.size, tick + radius + 1)].min())
            if tick + 1 < signal.size
            else 0.0
        )
        prominence = value_f - max(left_min, right_min)
        if is_rise or prominence >= prominence_th:
            candidates.append(tick)

    # 強いピークを優先する non-maximum suppression。
    selected: list[int] = []
    for tick in sorted(candidates, key=lambda t: float(signal[t]), reverse=True):
        if all(abs(tick - other) >= radius for other in selected):
            selected.append(tick)
    return sorted(selected)


def densify_onsets_on_plateau(
    envelope: np.ndarray,
    onsets: list[int],
    *,
    relative_threshold: float = 0.2,
    min_distance: int = 2,
) -> list[int]:
    """活性帯が続く平坦領域に、局所最大ベースの onset を補充する。

    ピーク検出だけでは平坦な U-Net 出力で疎になりすぎるため、
    envelope が閾値以上の区間だけを歩き、各 ``min_distance`` 窓で
    既存ピークまたは argmax を共通 onset にする。
    """
    signal = np.asarray(envelope, dtype=np.float32)
    if signal.size == 0:
        return []

    peak_max = float(signal.max())
    if peak_max <= 0.0:
        return sorted(onsets)

    active = signal >= peak_max * float(relative_threshold)
    radius = max(1, int(min_distance))
    chosen: set[int] = set(int(t) for t in onsets if 0 <= int(t) < signal.size)

    tick = 0
    while tick < signal.size:
        if not bool(active[tick]):
            tick += 1
            continue
        run_end = tick
        while run_end < signal.size and bool(active[run_end]):
            run_end += 1

        pos = tick
        while pos < run_end:
            window_end = min(pos + radius, run_end)
            existing = [t for t in chosen if pos <= t < window_end]
            if existing:
                pick = min(existing)
            else:
                local = signal[pos:window_end]
                pick = pos + int(np.argmax(local))
                chosen.add(pick)
            pos = pick + radius
        tick = run_end

    return sorted(chosen)


def decode_chord_peak_pianoroll(
    values: np.ndarray,
    *,
    category: int,
    pitch_min: int,
    pitch_max: int,
    onset_th: float = 0.3,
    peak_relative_threshold: float = 0.2,
    peak_min_distance: int = 2,
    peak_relative_prominence: float = 0.01,
    peak_rise_relative: float = 0.04,
    max_voices: int = 6,
    release_ratio: float = 0.55,
    release_gap_ticks: int = 1,
    minimum_peaks: int | None = None,
) -> np.ndarray | None:
    """連続 CHW 出力を、コード共通 onset の 0/1/2 ロールへ変換する。

    指定音域の上位 ``max_voices`` 個を足した時間 envelope から共通 onset
    を検出する（B: ピーク／立ち上がり／活性帯の局所最大、C: 複数 pitch 共有）。
    各 onset ではその近傍で ``onset_th`` 以上のピッチだけを採用し、次の共通
    onset または活性低下までを sustain (2) とする。次の onset との間には
    ``release_gap_ticks`` 分の無音を残し、打ち直しが被って聞こえるのを防ぐ。

    意味のあるピーク／ノートを作れない場合は ``None`` を返す。
    ``minimum_peaks`` 未指定時はパッチ内で最低 4 個を要求する。
    """
    raw = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if raw.ndim != 3:
        raise ValueError(f"values must be CHW, got shape={raw.shape}")
    if not 0 <= category < raw.shape[0]:
        raise ValueError(f"category out of range: {category}")

    lo = max(0, int(pitch_min))
    hi = min(raw.shape[2] - 1, int(pitch_max))
    if lo > hi:
        raise ValueError(f"invalid pitch range: {pitch_min}-{pitch_max}")

    roll = np.clip(raw[category, :, lo : hi + 1], 0.0, None)
    voice_count = max(1, min(int(max_voices), roll.shape[1]))
    if voice_count < roll.shape[1]:
        top = np.partition(roll, -voice_count, axis=1)[:, -voice_count:]
    else:
        top = roll
    envelope = top.sum(axis=1)
    onsets = find_temporal_peaks(
        envelope,
        relative_threshold=peak_relative_threshold,
        min_distance=peak_min_distance,
        relative_prominence=peak_relative_prominence,
        rise_relative=peak_rise_relative,
    )
    onsets = densify_onsets_on_plateau(
        envelope,
        onsets,
        relative_threshold=peak_relative_threshold,
        min_distance=peak_min_distance,
    )
    required_peaks = (
        max(4, raw.shape[1] // (TICKS_PER_BAR * 2))
        if minimum_peaks is None
        else max(1, int(minimum_peaks))
    )
    if len(onsets) < required_peaks:
        return None

    quantized = np.zeros_like(raw, dtype=np.int32)
    emitted = 0
    for onset_index, onset in enumerate(onsets):
        next_onset = (
            onsets[onset_index + 1]
            if onset_index + 1 < len(onsets)
            else raw.shape[1]
        )
        window_start = max(0, onset - 1)
        window_end = min(raw.shape[1], onset + 2)
        local_strength = roll[window_start:window_end].max(axis=0)
        active_local = np.flatnonzero(local_strength >= float(onset_th))
        if active_local.size == 0:
            continue

        if active_local.size > max_voices:
            strongest = np.argsort(local_strength[active_local])[-max_voices:]
            active_local = active_local[strongest]

        for local_pitch in active_local:
            pitch = lo + int(local_pitch)
            track = roll[:, int(local_pitch)]
            quantized[category, onset, pitch] = 1

            release_th = max(
                float(onset_th) * 0.5,
                float(local_strength[local_pitch]) * float(release_ratio),
            )
            sustain_limit = max(onset + 1, next_onset - max(0, int(release_gap_ticks)))
            end = onset + 1
            while end < sustain_limit and float(track[end]) >= release_th:
                quantized[category, end, pitch] = 2
                end += 1
            emitted += 1

    return quantized if emitted else None


def constrain_category_to_input_onsets(
    predicted_chw: np.ndarray,
    input_chw: np.ndarray,
    category: int,
) -> np.ndarray:
    """入力の発音点に対応する音だけを残し、それ以外の予測オンセットを除去する。"""
    out = np.zeros_like(predicted_chw)
    input_cat = input_chw[category]
    pred_cat = predicted_chw[category]

    for pitch in range(predicted_chw.shape[2]):
        for time in range(predicted_chw.shape[1]):
            if input_cat[time, pitch] != 1:
                continue
            t = time
            while t < predicted_chw.shape[1] and pred_cat[t, pitch] > 0:
                out[category, t, pitch] = pred_cat[t, pitch]
                t += 1

    return out


def constrain_guitar_from_input_onsets(
    raw_chw: np.ndarray,
    input_chw: np.ndarray,
    category: int,
    *,
    sustain_threshold: float = 0.35,
) -> np.ndarray:
    """入力オンセットを固定し、音価だけモデル出力から決める。"""
    out = np.zeros(raw_chw.shape, dtype=np.int32)
    input_cat = input_chw[category]
    pred_cat = raw_chw[category]

    for pitch in range(raw_chw.shape[2]):
        for time in range(raw_chw.shape[1]):
            if input_cat[time, pitch] != 1:
                continue
            out[category, time, pitch] = 1
            t = time + 1
            while t < raw_chw.shape[1] and pred_cat[t, pitch] >= sustain_threshold:
                out[category, t, pitch] = 2 if pred_cat[t, pitch] >= 0.75 else 1
                t += 1

    return out


def filter_pitch_range_chw(
    tonal_chw: np.ndarray,
    *,
    pitch_min: int,
    pitch_max: int,
) -> np.ndarray:
    """指定音域外のピッチをゼロにする。"""
    filtered = tonal_chw.copy()
    if pitch_min > 0:
        filtered[:, :, :pitch_min] = 0
    if pitch_max < 127:
        filtered[:, :, pitch_max + 1 :] = 0
    return filtered


def save_patches(patches: list[MidiPatch], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for patch in patches:
        stem = f"bar{patch.bar_index:04d}"
        np.save(output_dir / f"{stem}_tonal.npy", patch.tonal_chw)
        np.save(output_dir / f"{stem}_drum.npy", patch.drum_chw)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    midi_path = script_dir / "midi" / "backing_marusa_C_bpm100_8bars_skeleton.mid"
    out_dir = script_dir / "data" / "patches" / "backing_demo"

    patch_list = midi_to_patches(midi_path)
    save_patches(patch_list, out_dir)
    first = patch_list[0]
    print(f"パッチ数: {len(patch_list)}")
    print(f"tonal shape: {first.tonal_chw.shape}, drum shape: {first.drum_chw.shape}")
    print(f"保存先: {out_dir}")
