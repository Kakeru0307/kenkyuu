"""makeData 共通定数。"""

from __future__ import annotations

DEFAULT_RESOLUTION = 4
TICKS_PER_BAR = 16
BEAT_TICKS = 4
DEFAULT_BARS = 8
DEFAULT_VELOCITY = 80

# パワーコードを含む割合（発音／フレーズ単位）
BACKING_POWER_CHORD_PROBABILITY = 0.70
LEAD_POWER_CHORD_PHRASE_PROBABILITY = 0.40
# solo は未実装。将来のデータ生成で使う確定方針。
SOLO_POWER_CHORD_PHRASE_PROBABILITY = 0.20

KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

KEY_TO_SEMITONE = {
    "C": 0,
    "Db": 1,
    "D": 2,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "Gb": 6,
    "G": 7,
    "Ab": 8,
    "A": 9,
    "Bb": 10,
    "B": 11,
}

CHORD_QUALITIES = ("maj", "min", "7", "m7")

CHORD_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
}

SCALE_MODES = ("major", "natural_minor")

SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
}

# 旧: 単一コード／スケール往復。新規生成の主戦力は progression_*
PATTERN_TYPES = (
    "progression_strum",
    "progression_arpeggio",
    "chord_strum",
    "arpeggio",
    "scale_up",
    "scale_down",
)

# バッキングはストローク専用（arp/scale は 0。総量は SYNTHETIC_COUNT のまま）
PATTERN_WEIGHTS: dict[str, float] = {
    "progression_strum": 1.0,
    "progression_arpeggio": 0.0,
    "chord_strum": 0.0,
    "arpeggio": 0.0,
    "scale_up": 0.0,
    "scale_down": 0.0,
}

# バッキング・ストロークの音価型（1曲=1ノリ）
STRUM_ARTICULATIONS = (
    "solid",
    "staccato",
    "mixed",
    "sustained",
    "rests",
)

STRUM_ARTICULATION_WEIGHTS: dict[str, float] = {
    "solid": 0.30,
    "staccato": 0.30,
    "mixed": 0.25,
    "sustained": 0.10,
    "rests": 0.05,
}

# 1小節あたりの発音回数 N（実験: 8分刻み固定。他は変えない）
ATTACKS_PER_BAR_WEIGHTS: dict[int, float] = {
    8: 1.0,
}

PLACEMENT_TYPES = (
    "even",
    "front",
    "back",
    "offbeat",
    "sparse_random",
)

PLACEMENT_WEIGHTS: dict[str, float] = {
    "even": 1.0,
    "front": 0.0,
    "back": 0.0,
    "offbeat": 0.0,
    "sparse_random": 0.0,
}

# テスト: 2000（even固定で刻みを確認したら 15000 に戻す）
DEFAULT_SYNTHETIC_COUNT = 2000

BPM_RANGE = (60, 150)

# 既定は 8 小節のみ（16/24/32 は 1 MIDI から多数パッチが出てディスクを圧迫する）
BAR_LENGTH_CHOICES = (8,)

# ギターらしい音域（MIDI note number）
GUITAR_PITCH_MIN = 40
GUITAR_PITCH_MAX = 76

# ベース音域・プログラム（GM Electric Bass / finger → category 4）
BASS_PITCH_MIN = 28
BASS_PITCH_MAX = 55
BASS_PROGRAM = 33

# GM Drum Kit pitches
DRUM_KICK = 36
DRUM_SIDE_STICK = 37
DRUM_SNARE = 38
DRUM_SNARE_E = 40
DRUM_FLOOR_TOM_L = 41
DRUM_CLOSED_HAT = 42
DRUM_FLOOR_TOM_H = 43
DRUM_PEDAL_HAT = 44
DRUM_TOM_L = 45
DRUM_OPEN_HAT = 46
DRUM_TOM_LM = 47
DRUM_TOM_HM = 48
DRUM_CRASH_1 = 49
DRUM_TOM_H = 50
DRUM_RIDE = 51
DRUM_CRASH_2 = 57

DRUM_TOM_FILLS = (
    DRUM_TOM_H,
    DRUM_TOM_HM,
    DRUM_TOM_LM,
    DRUM_TOM_L,
    DRUM_FLOOR_TOM_H,
    DRUM_FLOOR_TOM_L,
)

# ドラム型カタログ（骨格固定・装飾のみ変動）。U-Net 条件 one-hot の次元順。
BEAT_TYPES = (
    "eight_basic",
    "four_floor",
    "sixteen_basic",
    "sixteen_funk",
    "halftime",
    "halftime_shuffle",
    "ballad_sparse",
    "shuffle_eight",
    "disco",
    "tresillo",
    "reggae",
    "metal_double",
)

# 型ごとの自然な BPM 帯（教師生成で共起を保証）
BEAT_BPM_RANGE: dict[str, tuple[int, int]] = {
    "ballad_sparse": (60, 95),
    "reggae": (60, 100),
    "halftime_shuffle": (70, 110),
    "shuffle_eight": (70, 120),
    "sixteen_funk": (70, 120),
    "halftime": (80, 120),
    "sixteen_basic": (80, 120),
    "eight_basic": (80, 140),
    "tresillo": (80, 130),
    "disco": (110, 135),
    "four_floor": (110, 150),
    "metal_double": (120, 150),
}
