import pretty_midi
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

def midi_to_vitex_image(midi_path, output_path, pixels_per_second=20):
    # 1. MIDIファイルの読み込み
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # 2. キャンバスサイズの計算
    # 横幅: 曲の総時間(秒) × 1秒あたりのピクセル数
    max_time = midi_data.get_end_time()
    width = int(max_time * pixels_per_second) + 1
    height = 128  # MIDIノート番号(0-127)
    
    # 真っ黒なRGB画像（高さ, 幅, 3チャンネル）を作成
    # データ型は画像として保存しやすい8ビット符号なし整数(0-255)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # U-Netに入力するための一定の明るさ（ベロシティ64相当）
    flat_brightness = 128 
    
    # 3. トラックごとの描画処理
    for instrument in midi_data.instruments:
        # ドラムトラックは除外
        if instrument.is_drum:
            continue
            
        # プログラム番号等で楽器を判別し、RGBチャンネルを割り当てる（簡易的な例）
        # R=0(メロディ想定), G=1(伴奏想定), B=2(ベース想定)
        channel = 0
        if instrument.program > 20: # ギターやピアノ以外を伴奏扱いにする等のロジック
            channel = 1
        if instrument.program > 32: # ベース系のプログラム番号
            channel = 2

        # 4. 音符を直線として描画
        for note in instrument.notes:
            # 時間をピクセル座標（X軸）に変換
            x_start = int(note.start * pixels_per_second)
            x_end = int(note.end * pixels_per_second)
            
            # ピッチをY軸に変換（画像は上がY=0なので、高音が上になるよう反転）
            y = 127 - note.pitch
            
            # 該当する座標をフラットな明るさで塗りつぶす
            canvas[y, x_start:x_end, channel] = flat_brightness

    # 5. 画像として保存
    img = Image.fromarray(canvas)
    img.save(output_path)
    print(f"画像を保存しました: {output_path} (幅: {width}px, 高さ: 128px)")

if __name__ == "__main__":
    midi_to_vitex_image(
        SCRIPT_DIR / "midi" / "test1.mid",
        SCRIPT_DIR / "vitex_input.png",
    )