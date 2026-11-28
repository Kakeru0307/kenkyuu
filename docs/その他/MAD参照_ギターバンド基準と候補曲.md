# MAD参照 — ギター前景インスト基準と現行セット

更新: 2026-11-21  
採用は目視（曲同定＋MIDI中身。スコア機械・一括自動採用は禁止）。

---

## 0. 現状

| 項目 | 数 |
|---|---|
| `testMad/ref/midi` | **101** |
| `testMad/ref/wav` | **101**（Fluid R3 GM） |
| 生成側 | `testMad/gen` 100本 |

Drive の `ref/wav` を上げ直し、Colab は `CLEAR_REF_ONLY` で再計算。  
追加曲のチャート／認定の具体値: `docs/その他/MAD参照_追加曲の人気指標.md`

---

## 1. 基準（広げた版）

| # | 基準 | 内容 |
|---|---|---|
| A | 前景 | ギター前景を優先。オルガン／ファンク／フュージョンの**インスト定番**も可 |
| B | 編成 | バンド感（ドラム or 明確ビート）。アコギ小品も可 |
| C | 長さ | おおよそ 40秒〜6.5分（極端な長尺は除外） |
| D | 除外 | ボーカル曲、ディスコ／映画オーケストラ主役、スムース・サックス主役のみ、ゲームMIDI誤ヒット |
| E | 取得 | clean_midi / BitMIDI 個別。同一 SF で WAV |

---

## 2. 今回追加した主な系統

- ギターヒーロー: Satriani 追加、Malmsteen Crying、Vai 既存、Cliffs、The Loner、Eruption/Cathedral/316
- ロック・インスト: Rush YYZ、Zeppelin Moby Dick / Black Mountain Side、Yes Clap / Mood for a Day、Metallica Anesthesia、Deep Purple Fireball、King Crimson Discipline
- バンド定番インスト: Booker T Green Onions / Hip Hug-Her、AWB Pick Up the Pieces、Santana Europa、Ventures 追加、Link Wray Rumble、Spotnicks
- フュージョン: Zappa Peaches / Black Page 等、Pat Metheny、Lee Ritenour

---

## 3. 現行ファイル一覧（101）

```
003_The_Champs__Tequila.mid
005_Duane_Eddy__Rebel_Rouser.mid
006_Duane_Eddy__Cannonball.mid
007_The_Rockin_R_s__The_Beat.mid
008_The_Virtues__Guitar_Boogie_Shuffle.mid
010_Johnny_and_the_Hurricanes__Crossfire.mid
013_Johnny_and_the_Hurricanes__Red_River_Rock.mid
017_The_Fireballs__Bulldog.mid
020_Duane_Eddy__Peter_Gunn.mid
022_The_Shadows__Apache.mid
023_The_Ventures__Walk_Don_t_Run.mid
028_The_Shadows__The_Savage.mid
030_The_Champs__Limbo_Rock.mid
031_Jet_Harris__Besame_Mucho.mid
032_The_Shadows__Wonderful_Land.mid
033_The_Shadows__Guitar_Tango.mid
034_The_Shadows__Dance_On_.mid
037_The_Dakotas__The_Cruel_Sea.mid
038_Lonnie_Mack__Memphis.mid
039_Lonnie_Mack__Wham_.mid
042_The_Shadows__Foot_Tapper.mid
043_The_Surfaris__Wipe_Out.mid
044_The_Pyramids__Penetration.mid
055_Edgar_Winter_Group__Frankenstein.mid
060_Focus__Hocus_Pocus.mid
A_Johnny_And_The_Hurricanes__Reveille_Rock.mid
Average_White_Band__Pick_Up_the_Pieces.mid
Booker_T_and_the_MGs__Green_Onions.mid
Booker_T_and_the_MGs__Hip_Hug-Her.mid
C_Gary_Glitter__Rock_and_Roll_Part_2.mid
Deep_Purple__Fireball.mid
Dick_Dale__Misirlou.mid
E_Fleetwood_Mac__Albatross.mid
E_Santo_Johnny__Sleep_Walk.mid
E_The_Shadows__Atlantis.mid
E_The_Shadows__F.B.I..mid
E_The_Shadows__Kon-Tiki.mid
E_The_Ventures__Pipeline.mid
E2_The_Shadows__Guitar_Boogie.mid
E2_The_Shadows__Riders_in_the_Sky.mid
E2_The_Shadows__Walk_Dont_Run.mid
E2_The_Ventures__Apache.mid
E2_The_Ventures__Blue_Star.mid
E2_The_Ventures__Bulldog.mid
E2_The_Ventures__Diamond_Head.mid
E2_The_Ventures__Honky_Tonk.mid
Eric_Johnson__Cliffs_of_Dover.mid
Frank_Zappa__Peaches_en_Regalia.mid
Frank_Zappa__The_Black_Page_1.mid
Frank_Zappa__The_Girl_in_the_Magnesium_Dress.mid
Frank_Zappa__The_Little_House_I_Used_to_Live_In.mid
Frank_Zappa__Uncle_Meat.mid
Gary_Moore__The_Loner.mid
Joe_Satriani__Always_With_Me_Always_With_You.mid
Joe_Satriani__Crushing_Day.mid
Joe_Satriani__Cryin.mid
Joe_Satriani__Day_at_the_Beach.mid
Joe_Satriani__Summer_Song.mid
Joe_Satriani__Surfing_With_the_Alien.mid
Joe_Satriani__The_Mighty_Turtle_Head.mid
King_Crimson__Discipline.mid
Led_Zeppelin__Black_Mountain_Side.mid
Led_Zeppelin__Moby_Dick.mid
Lee_Ritenour__Night_Rhythms.mid
Link_Wray__Rumble.mid
Metallica__Anesthesia_Pulling_Teeth.mid
Pat_Metheny__American_Garage.mid
Pat_Metheny__Are_You_Going_With_Me.mid
Rush__YYZ.mid
Santana__Europa.mid
Santana__Samba_Pa_Ti.mid
Steve_Vai__Sisters.mid
The_Allman_Brothers_Band__In_Memory_of_Elizabeth_Reed.mid
The_Shadows__Quartermaster_Stores.mid
The_Spotnicks__Amapola.mid
The_Spotnicks__Happy_Guitar.mid
The_Ventures__Besame_Mucho.mid
The_Ventures__Bird_Rockers.mid
The_Ventures__Bluebird.mid
The_Ventures__Eleventh_Hour.mid
The_Ventures__Escape.mid
The_Ventures__Fugitive.mid
The_Ventures__Gringo.mid
The_Ventures__Guitar_Boogie_Shuffle.mid
The_Ventures__Guitar_Twist.mid
The_Ventures__I_Feel_Fine.mid
The_Ventures__Instant_Guitar.mid
The_Ventures__Jokers_Wild.mid
The_Ventures__Penetration.mid
The_Ventures__The_Cruel_Sea.mid
The_Ventures__The_Good_The_Bad_and_The_Ugly.mid
The_Ventures__Theme_From_Come_September.mid
Van_Halen__316.mid
Van_Halen__Cathedral.mid
Van_Halen__Eruption.mid
W_Dennis_Coffey__Scorpio.mid
W_Duane_Eddy__Pepe.mid
Yes__Clap.mid
Yes__Mood_for_a_Day.mid
Yngwie_Malmsteen__Crying.mid
Yngwie_Malmsteen__Leviathan.mid

```
