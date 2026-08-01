"""典型コード進行の定義と、キー上のコード列への展開。"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import KEY_TO_SEMITONE
from .voicings import chord_pitches

# 度数: 1=I ... 7=VII。accidentals: -1=♭, +1=♯
# quality は maj/min/7/m7/dim
Degree = tuple[int, int, str]  # (scale_degree 1-7, accidental, quality)


@dataclass(frozen=True)
class ProgressionSpec:
    name: str
    family: str  # diatonic_major | diatonic_minor | borrowed | blues
    mode: str  # major | natural_minor（生成時のスケール前提）
    degrees: tuple[Degree, ...]
    weight: float = 1.0


def _d(degree: int, quality: str, accidental: int = 0) -> Degree:
    return (degree, accidental, quality)


# --- Major diatonic ---
PROGRESSIONS: tuple[ProgressionSpec, ...] = (
    ProgressionSpec(
        "marusa",
        "diatonic_major",
        "major",
        (_d(4, "maj"), _d(5, "maj"), _d(3, "min"), _d(6, "min")),
        weight=2.0,
    ),
    ProgressionSpec(
        "komuro",
        "diatonic_major",
        "major",
        (_d(6, "min"), _d(4, "maj"), _d(5, "maj"), _d(1, "maj")),
        weight=2.0,
    ),
    ProgressionSpec(
        "canon_short",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(5, "maj"), _d(6, "min"), _d(4, "maj")),
        weight=2.0,
    ),
    ProgressionSpec(
        "canon_full",
        "diatonic_major",
        "major",
        (
            _d(1, "maj"),
            _d(5, "maj"),
            _d(6, "min"),
            _d(3, "min"),
            _d(4, "maj"),
            _d(1, "maj"),
            _d(4, "maj"),
            _d(5, "maj"),
        ),
        weight=1.5,
    ),
    ProgressionSpec(
        "jpop_subdom",
        "diatonic_major",
        "major",
        (_d(4, "maj"), _d(5, "maj"), _d(6, "min"), _d(3, "min")),
        weight=1.5,
    ),
    ProgressionSpec(
        "classic_turnaround",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(4, "maj"), _d(5, "maj"), _d(1, "maj")),
        weight=1.5,
    ),
    ProgressionSpec(
        "two_five_one",
        "diatonic_major",
        "major",
        (_d(2, "min"), _d(5, "maj"), _d(1, "maj"), _d(1, "maj")),
        weight=1.2,
    ),
    ProgressionSpec(
        "doowop",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(6, "min"), _d(4, "maj"), _d(5, "maj")),
        weight=1.5,
    ),
    ProgressionSpec(
        "pop_axis",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(5, "maj"), _d(6, "min"), _d(4, "maj")),
        weight=1.5,
    ),
    ProgressionSpec(
        "cycle_1625",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(6, "min"), _d(2, "min"), _d(5, "maj")),
        weight=1.2,
    ),
    ProgressionSpec(
        "subdom_start",
        "diatonic_major",
        "major",
        (_d(4, "maj"), _d(1, "maj"), _d(5, "maj"), _d(6, "min")),
        weight=1.2,
    ),
    ProgressionSpec(
        "resolve_4516",
        "diatonic_major",
        "major",
        (_d(4, "maj"), _d(5, "maj"), _d(1, "maj"), _d(6, "min")),
        weight=1.2,
    ),
    ProgressionSpec(
        "descending_bass",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(5, "maj"), _d(6, "min"), _d(5, "maj")),
        weight=1.2,
    ),
    ProgressionSpec(
        "deceptive",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(5, "maj"), _d(6, "min"), _d(6, "min")),
        weight=1.0,
    ),
    ProgressionSpec(
        "plagal_ish",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(4, "maj"), _d(1, "maj"), _d(5, "maj")),
        weight=1.0,
    ),
    ProgressionSpec(
        "simple_15",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(5, "maj"), _d(1, "maj"), _d(5, "maj")),
        weight=0.8,
    ),
    ProgressionSpec(
        "simple_14",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(4, "maj"), _d(1, "maj"), _d(4, "maj")),
        weight=0.8,
    ),
    ProgressionSpec(
        "simple_16",
        "diatonic_major",
        "major",
        (_d(1, "maj"), _d(6, "min"), _d(1, "maj"), _d(6, "min")),
        weight=0.8,
    ),
    ProgressionSpec(
        "simple_64",
        "diatonic_major",
        "major",
        (_d(6, "min"), _d(4, "maj"), _d(6, "min"), _d(4, "maj")),
        weight=0.8,
    ),
    ProgressionSpec(
        "simple_45",
        "diatonic_major",
        "major",
        (_d(4, "maj"), _d(5, "maj"), _d(4, "maj"), _d(5, "maj")),
        weight=0.8,
    ),
    # --- Minor diatonic ---
    ProgressionSpec(
        "minor_komuro",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(6, "maj"), _d(3, "maj"), _d(7, "maj")),
        weight=2.0,
    ),
    ProgressionSpec(
        "minor_natural_loop",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(7, "maj"), _d(6, "maj"), _d(7, "maj")),
        weight=1.8,
    ),
    ProgressionSpec(
        "minor_basic",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(4, "min"), _d(5, "min"), _d(1, "min")),
        weight=1.5,
    ),
    ProgressionSpec(
        "minor_rock",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(6, "maj"), _d(7, "maj"), _d(1, "min")),
        weight=1.8,
    ),
    ProgressionSpec(
        "minor_expand",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(4, "min"), _d(7, "maj"), _d(3, "maj")),
        weight=1.3,
    ),
    ProgressionSpec(
        "minor_marusa_like",
        "diatonic_minor",
        "natural_minor",
        (_d(6, "maj"), _d(7, "maj"), _d(5, "min"), _d(1, "min")),
        weight=1.5,
    ),
    ProgressionSpec(
        "minor_simple_17",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(7, "maj"), _d(1, "min"), _d(7, "maj")),
        weight=0.8,
    ),
    ProgressionSpec(
        "minor_simple_16",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(6, "maj"), _d(1, "min"), _d(6, "maj")),
        weight=0.8,
    ),
    ProgressionSpec(
        "minor_simple_14",
        "diatonic_minor",
        "natural_minor",
        (_d(1, "min"), _d(4, "min"), _d(1, "min"), _d(4, "min")),
        weight=0.8,
    ),
    # --- Borrowed / rock ---
    ProgressionSpec(
        "rock_bVII",
        "borrowed",
        "major",
        (_d(1, "maj"), _d(7, "maj", -1), _d(4, "maj"), _d(1, "maj")),
        weight=1.2,
    ),
    ProgressionSpec(
        "rock_I_IV_bVII_IV",
        "borrowed",
        "major",
        (_d(1, "maj"), _d(4, "maj"), _d(7, "maj", -1), _d(4, "maj")),
        weight=1.0,
    ),
    ProgressionSpec(
        "rock_vi_walkdown",
        "borrowed",
        "major",
        (_d(6, "min"), _d(5, "maj"), _d(4, "maj"), _d(5, "maj")),
        weight=1.0,
    ),
    # --- Blues (dominant) ---
    ProgressionSpec(
        "blues_12bar_short",
        "blues",
        "major",
        (
            _d(1, "7"),
            _d(1, "7"),
            _d(1, "7"),
            _d(1, "7"),
            _d(4, "7"),
            _d(4, "7"),
            _d(1, "7"),
            _d(1, "7"),
            _d(5, "7"),
            _d(4, "7"),
            _d(1, "7"),
            _d(5, "7"),
        ),
        weight=0.8,
    ),
)


PROGRESSION_BY_NAME = {spec.name: spec for spec in PROGRESSIONS}


def degree_root_key(tonic: str, degree: int, accidental: int = 0) -> str:
    """トニックキーから度数＋臨時記号のルート名を返す。"""
    tonic_pc = KEY_TO_SEMITONE[tonic]
    # メジャースケール上の度数オフセット（1-indexed）
    major_offsets = (0, 2, 4, 5, 7, 9, 11)
    pc = (tonic_pc + major_offsets[degree - 1] + accidental) % 12
    for name, value in KEY_TO_SEMITONE.items():
        if value == pc:
            return name
    return tonic


def minor_tonic_as_relative(major_tonic: str) -> str:
    """長調キー名から平行短調のトニック名を返す（VI = major tonic の相対）。"""
    # A minor の平行長調は C → 短調トニックは major の 6 度
    return degree_root_key(major_tonic, 6)


def resolve_progression_chords(
    spec: ProgressionSpec,
    key: str,
) -> list[tuple[str, str]]:
    """進行を (chord_root_key, quality) の列に展開する。

    - major 系: ``key`` を長調トニックとして解釈
    - natural_minor 系: ``key`` を短調トニックとして解釈
      （内部では平行長調の度数で計算するため、短調 i を major の vi 相当に写す）
    """
    chords: list[tuple[str, str]] = []

    if spec.mode == "natural_minor":
        # key = Am のとき、平行長調は C。度数は「短調の i=1」を
        # 平行長調の vi=6 にオフセットしてから major 度数表で解決する。
        minor_tonic_pc = KEY_TO_SEMITONE[key]
        relative_major_pc = (minor_tonic_pc + 3) % 12  # Am → C
        relative_major = next(n for n, v in KEY_TO_SEMITONE.items() if v == relative_major_pc)
        # 短調度数 d → 平行長調での度数: i→6, ii→7, III→1, iv→2, v→3, VI→4, VII→5
        minor_to_rel_major = {1: 6, 2: 7, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        for degree, accidental, quality in spec.degrees:
            rel_degree = minor_to_rel_major[degree]
            root = degree_root_key(relative_major, rel_degree, accidental)
            chords.append((root, quality))
        return chords

    for degree, accidental, quality in spec.degrees:
        root = degree_root_key(key, degree, accidental)
        chords.append((root, quality))
    return chords


def progression_chord_pitch_sets(
    spec: ProgressionSpec,
    key: str,
) -> list[list[int]]:
    return [
        chord_pitches(root, quality)
        for root, quality in resolve_progression_chords(spec, key)
    ]


def list_progression_summary() -> list[dict]:
    rows: list[dict] = []
    for spec in PROGRESSIONS:
        example_key = "C" if spec.mode == "major" else "A"
        chords = resolve_progression_chords(spec, example_key)
        label = "-".join(f"{r}{'' if q == 'maj' else q}" for r, q in chords)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "mode": spec.mode,
                "example_key": example_key,
                "chords": label,
                "bars_in_loop": len(spec.degrees),
                "weight": spec.weight,
            }
        )
    return rows
