"""MIDI program / ViTex category の対応（方針 B: ギターは category 3）。"""

from __future__ import annotations

import muspy

# GM: Electric Guitar (clean) → program // 8 == 3
GUITAR_PROGRAM = 27
GUITAR_OVERDRIVE_PROGRAM = 29
GUITAR_DISTORTION_PROGRAM = 30

GUITAR_PROGRAM_NAMES: dict[int, str] = {
    27: "Electric Guitar (clean)",
    29: "Overdriven Guitar",
    30: "Distortion Guitar",
}

# ギター想定音域（MIDI note number）
GUITAR_PITCH_MIN = 40
GUITAR_PITCH_MAX = 76

# GM Electric Bass (finger) → program // 8 == 4
BASS_PROGRAM = 33

# パッチ category → DAW 再生用の代表 program（未指定は category * 8）
CATEGORY_DEFAULT_PROGRAM: dict[int, int] = {
    3: GUITAR_PROGRAM,
    4: BASS_PROGRAM,
}


def guitar_track_name(program: int) -> str:
    return GUITAR_PROGRAM_NAMES.get(program, f"Guitar (program {program})")


def program_for_category(category: int) -> int:
    return CATEGORY_DEFAULT_PROGRAM.get(category, category * 8)


def remap_tonal_program(music: muspy.Music, program: int = GUITAR_PROGRAM) -> muspy.Music:
    """非ドラムトラックの program を統一する（Guitar-TECHS 用）。"""
    for track in music.tracks:
        if not track.is_drum:
            track.program = program
    return music
