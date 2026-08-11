"""アーティキュレーション・レイヤ。

U-Net/CVAE が生成した muspy.Music に対して velocity 調整・
pitch bend 生成を後付けで適用する。

technique_type は SongForm が決定済みの値を直接受け取る（TechniqueHead による推論は廃止）。
VelocityMLP が None の場合はパススルー。
"""

from __future__ import annotations

import muspy

from makeData.constants import technique_type_to_id
from velocity_mlp import VelocityMLP

GUITAR_PROGRAM_MIN = 24
GUITAR_PROGRAM_MAX = 31


def apply_articulation(
    music: muspy.Music,
    *,
    section_energy: float = 0.5,
    beat_type: str = "eight_basic",
    technique_type: str = "normal",
    bpm: float = 120.0,
    velocity_model: object | None = None,
    bend_model: object | None = None,
) -> muspy.Music:
    """リードギターパートにアーティキュレーションを適用する。

    全モデルが None の場合は入力をそのまま返す（パススルー）。

    Parameters
    ----------
    music:
        セクション単位で生成された muspy.Music。
    section_energy:
        区間のエネルギー（0.0〜1.0）。将来フルソング velocity アノテーションが
        揃えば学習に使えるが、現時点では VelocityMLP 入力には含めない。
    beat_type:
        ドラムパターン種別（"eight_basic" など）。将来の bend モデルで使用予定。
    technique_type:
        SongForm から決定済みのテクニック種別（"normal", "vibrato" など）。
    bpm:
        テンポ。
    velocity_model:
        VelocityMLP インスタンス。None のときは velocity を変更しない。
    bend_model:
        pitch bend カーブ生成モデル（Phase 3 で実装予定）。
    """
    if velocity_model is None and bend_model is None:
        return music

    technique_id = technique_type_to_id(technique_type)

    if velocity_model is not None:
        music = _apply_velocity(music, velocity_model, technique_id)

    # Phase 3: bend_model が揃ったときに実装する
    # if bend_model is not None:
    #     bend_events = _generate_bend_events(music, bend_model, technique_id)
    #     music._bend_events = bend_events

    return music


def _apply_velocity(
    music: muspy.Music,
    velocity_model: VelocityMLP,
    technique_id: int,
) -> muspy.Music:
    """VelocityMLP でノートごとの velocity を設定する。"""
    if not isinstance(velocity_model, VelocityMLP):
        raise TypeError(
            f"velocity_model は VelocityMLP である必要があります: {type(velocity_model)}"
        )

    ticks_per_beat = 4  # プロジェクトの resolution（1拍=4tick）

    for track in music.tracks:
        if track.is_drum:
            continue
        if not (GUITAR_PROGRAM_MIN <= track.program <= GUITAR_PROGRAM_MAX):
            continue
        for note in track.notes:
            dur_beats = note.duration / ticks_per_beat
            note.velocity = velocity_model.predict(
                technique_id=technique_id,
                pitch=int(note.pitch),
                duration_beats=dur_beats,
            )

    return music
