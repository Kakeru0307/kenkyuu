# MAD参照MIDI — 採用基準と選曲一覧

作成日: 2026-11-21  
更新: 2026-11-21（長尺12本差し替え後）  
本文書: `docs/その他/MAD参照MIDI_採用基準と選曲一覧.md`  
見直し案（ギターバンド基準・残す／外す／新規候補）: `docs/その他/MAD参照_ギターバンド基準と候補曲.md`  

| 項目 | 現行 |
|---|---|
| MIDI | `testMad/ref/midi/` **101本**（詳細は `MAD参照_ギターバンド基準と候補曲.md`） |
| WAV | `testMad/ref/wav/` **101本**（同一 SF・Fluid R3 GM。git 外・Drive へ） |
| 生成側 | `testMad/gen/midi/`・`gen/wav/` 各 **100本** |
| 長さ | 参照 WAV 最大 ≈4.9分（5分超なし）。MAD はこれらの mid/wav を使う |

---

## 1. 何のための選曲か

MAD（参照分布 vs 生成分布）の**参照側**用。  
ロック／ポップ系インストを集め、同一 SoundFont で MIDI→WAV して使う。

---

## 2. 採用基準（最終）

機械スコアは使わない。次を**目視**で満たすものを採用。

| # | 基準 | 内容 |
|---|---|---|
| A | チャート根拠 | ピークが確認できる曲・同曲の知名度あるインスト版（一部は定番インストで弱） |
| B | ファイル同定 | パス／ファイル名からアーティストと曲名が分かる |
| C | 取得条件 | 中身確認後に個別取得（LMD-full 一括DLはしない） |
| D | 長さ | MAD/VRAM 都合で極端な長尺は短い MIDI に差し替え可 |

### チャート根拠のソース

1. Wikipedia [List of rock instrumentals](https://en.wikipedia.org/wiki/List_of_rock_instrumentals)  
2. Billboard 系 Top 100 Instrumentals  
3. UK Singles インスト #1 一覧  

### 取得元

| 略号 | 取得元 |
|---|---|
| clean | `clean_midi` / Lakh MIDI Dataset Clean 相当から個別取得 |
| bitmidi | BitMIDI（曲名一致のもの） |

---

## 3. 現行セットの注記

- 本番パスは **`testMad/ref/{midi,wav}`**（旧 `_tmp_mad_ref` は削除済み）。  
- FBI の dongrays 重複は入れていない。  
- **2026-11-21**: 5分超だった参照12本を短い MIDI に差し替え（OOM 回避・生成側との長さ偏り緩和）。  
- 差し替え後の一部（Ventures アルバム寄り等）はチャート根拠が弱め。  
- 生成100本は本線 `generate_form.py`（文→prior→song_form）。

### 外した12本（長尺）

`Tubular_Bells` / `The_Good_The_Bad_And_The_Ugly` / `Wonderland_By_Night` / `A_Taste_Of_Honey` / `Autobahn` / `Rise` / `Wheels` / `Petite_Fleur` / `Grazing_in_the_Grass` / `Also_Sprach_Zarathustra` / `The_Hustle` / `Express`

### 入れた12本（短尺）

`Floyd_Cramer__Root_Beer_Rag` / `Dick_Dale__Misirlou` / `The_Shadows__Quartermaster_Stores` / `The_Ventures__{Instant_Guitar,Escape,The_Cruel_Sea,Jokers_Wild,Fugitive,Bird_Rockers,Eleventh_Hour,Theme_From_Come_September,Gringo}`

---

## 4. 現行ファイル一覧（`testMad/ref/midi`）

### 4-1. 番号付き（初期 rock instrumentals 系）

| ファイル | 曲 | 根拠 |
|---|---|---|
| 003_The_Champs__Tequila.mid | The Champs – Tequila | US#1 |
| 005_Duane_Eddy__Rebel_Rouser.mid | Duane Eddy – Rebel Rouser | US#6 UK#8 |
| 006_Duane_Eddy__Cannonball.mid | Duane Eddy – Cannonball | US#15 UK#2 |
| 007_The_Rockin_R_s__The_Beat.mid | The Rockin' R's – The Beat | US#57 ※弱め |
| 008_The_Virtues__Guitar_Boogie_Shuffle.mid | The Virtues – Guitar Boogie Shuffle | US#5 |
| 009_Dave_Baby_Cortez__The_Happy_Organ.mid | Dave "Baby" Cortez – The Happy Organ | US#1 |
| 010_Johnny_and_the_Hurricanes__Crossfire.mid | Johnny & the Hurricanes – Crossfire | US#23 |
| 013_Johnny_and_the_Hurricanes__Red_River_Rock.mid | Red River Rock | US#5 UK#3 |
| 017_The_Fireballs__Bulldog.mid | The Fireballs – Bulldog | US#24 ※弱め |
| 018_Bill_Black_s_Combo__Don_t_Be_Cruel.mid | Bill Black's Combo – Don't Be Cruel | US#11 |
| 020_Duane_Eddy__Peter_Gunn.mid | Duane Eddy – Peter Gunn | US#8 UK#6 |
| 021_Floyd_Cramer__Last_Date.mid | Floyd Cramer – Last Date | US#2 |
| 022_The_Shadows__Apache.mid | The Shadows – Apache | UK#1 |
| 023_The_Ventures__Walk_Don_t_Run.mid | The Ventures – Walk, Don't Run | US#2 |
| 026_The_Mar-Keys__Last_Night.mid | The Mar-Keys – Last Night | US#3 |
| 028_The_Shadows__The_Savage.mid | The Shadows – The Savage | UK#10 |
| 030_The_Champs__Limbo_Rock.mid | The Champs – Limbo Rock | US#40 |
| 031_Jet_Harris__Besame_Mucho.mid | Jet Harris – Besame Mucho | UK#22 |
| 032_The_Shadows__Wonderful_Land.mid | The Shadows – Wonderful Land | UK#1 |
| 033_The_Shadows__Guitar_Tango.mid | The Shadows – Guitar Tango | UK#4 |
| 034_The_Shadows__Dance_On_.mid | The Shadows – Dance On! | UK#1 |
| 035_The_Tornados__Telstar.mid | The Tornados – Telstar | US#1 UK#1 |
| 036_Booker_T._the_M.G._s__Green_Onions.mid | Booker T. & the M.G.'s – Green Onions | US#3 |
| 037_The_Dakotas__The_Cruel_Sea.mid | The Dakotas – The Cruel Sea | UK#18 |
| 038_Lonnie_Mack__Memphis.mid | Lonnie Mack – Memphis | US#5 |
| 039_Lonnie_Mack__Wham_.mid | Lonnie Mack – Wham! | US#24 ※弱め |
| 042_The_Shadows__Foot_Tapper.mid | The Shadows – Foot Tapper | UK#1 |
| 043_The_Surfaris__Wipe_Out.mid | The Surfaris – Wipe Out | US#2 |
| 044_The_Pyramids__Penetration.mid | The Pyramids – Penetration | US#18 |
| 046_Cliff_Nobles_Co.__The_Horse.mid | Cliff Nobles & Co. – The Horse | US#2 |
| 049_Mason_Williams__Classical_Gas.mid | Mason Williams – Classical Gas | US#2 |
| 053_Hot_Butter__Popcorn.mid | Hot Butter – Popcorn | US#9 |
| 055_Edgar_Winter_Group__Frankenstein.mid | Edgar Winter Group – Frankenstein | US#1 |
| 056_The_Love_Unlimited_Orchestra__Love_s_Theme.mid | Love Unlimited Orchestra – Love's Theme | US#1 |
| 057_Average_White_Band__Pick_Up_the_Pieces.mid | Average White Band – Pick Up the Pieces | US#1 |
| 060_Focus__Hocus_Pocus.mid | Focus – Hocus Pocus | US#9 |
| 061_MFSB__TSOP_The_Sound_of_Philadelphia_.mid | MFSB – TSOP | US#1 |
| 062_Silver_Convention__Fly_Robin_Fly.mid | Silver Convention – Fly, Robin, Fly | US#1 |
| 064_Walter_Murphy__A_Fifth_Of_Beethoven.mid | Walter Murphy – A Fifth of Beethoven | US#1 |
| 065_Rhythm_Heritage__Theme_from_S.W.A.T..mid | Rhythm Heritage – Theme from S.W.A.T. | US#1 |
| 066_Chuck_Mangione__Feels_So_Good.mid | Chuck Mangione – Feels So Good | US#4 |

### 4-2. 拡充分（A/B/C/E/E2/W）

| ファイル | 曲 | 根拠／注 | 取得 |
|---|---|---|---|
| A_Johnny_And_The_Hurricanes__Reveille_Rock.mid | Reveille Rock | US#25 | clean |
| E_The_Shadows__Kon-Tiki.mid | Kon-Tiki | UK#1 | clean |
| E_The_Shadows__F.B.I..mid | F.B.I. | UK#6 | clean |
| E_Santo_Johnny__Sleep_Walk.mid | Sleep Walk（Shadows 演奏） | 原曲 US#1 | clean |
| E_Fleetwood_Mac__Albatross.mid | Albatross | UK#1 | clean |
| E_The_Ventures__Pipeline.mid | Pipeline | Chantays 等チャート曲のカバー | clean |
| E_Booker_T._the_M.G._s__Hip_Hug-Her.mid | Hip Hug-Her | Hot 100 #37 | clean |
| E_Herb_Alpert__The_Lonely_Bull.mid | The Lonely Bull | US#6 | clean |
| E_The_Shadows__Atlantis.mid | Atlantis | UK#2 | clean |
| W_Duane_Eddy__Pepe.mid | Pepe | US#18 UK#2 | bitmidi |
| W_Booker_T._the_M.G._s__Time_Is_Tight.mid | Time Is Tight | US#6 | bitmidi |
| W_Dennis_Coffey__Scorpio.mid | Scorpio | US#6 | bitmidi |
| C_Percy_Faith__Theme_From_A_Summer_Place.mid | Theme from A Summer Place | US#1 | clean |
| C_Paul_Mauriat__Love_Is_Blue.mid | Love Is Blue | US#1 | clean |
| C_Lawrence_Welk__Calcutta.mid | Calcutta（Ventures 演奏） | 原曲 US#1 | clean |
| C_Meco__Star_Wars_Theme.mid | Star Wars Theme | US#1 | clean |
| C_Acker_Bilk__Stranger_On_The_Shore.mid | Stranger on the Shore | US#1 | clean |
| C_Vangelis__Chariots_Of_Fire.mid | Chariots of Fire | US#1 | clean |
| C_Jan_Hammer__Miami_Vice_Theme.mid | Miami Vice Theme | US#1 | clean |
| C_Harold_Faltermeyer__Axel_F.mid | Axel F | US#3 | clean |
| C_Marvin_Hamlisch__The_Entertainer.mid | The Entertainer | US#3 | clean |
| C_Frank_Mills__Music_Box_Dancer.mid | Music Box Dancer | US#3 | clean |
| C_Gary_Glitter__Rock_and_Roll_Part_2.mid | Rock and Roll Part 2 | US#7 | clean |
| C_Spyro_Gyra__Morning_Dance.mid | Morning Dance | US#24 | clean |
| C_Herbie_Hancock__Chameleon.mid | Chameleon | US#42 | clean |
| C_B._Bumble_and_the_Stingers__Nut_Rocker.mid | Nut Rocker | UK#1 | clean |
| C_Kenny_G__Songbird.mid | Songbird | US#4 | clean |
| B_Perez_Prado__Cherry_Pink_and_Apple_Blossom_White.mid | Cherry Pink… | US#1 | bitmidi |
| B_Morris_Stoloff__Moonglow_and_Theme_From_Picnic.mid | Moonglow & Theme from Picnic | US#1 | bitmidi |
| B_David_Rose__The_Stripper.mid | The Stripper | US#1 | bitmidi |
| B_John_Williams__Theme_From_Jaws.mid | Theme from Jaws | Hot 100 / BB 70s | bitmidi |
| E2_Bert_Kaempfert__A_Swinging_Safari.mid | A Swinging Safari | Hot 100 | clean |
| E2_Henry_Mancini__Baby_Elephant_Walk.mid | Baby Elephant Walk | US#10 | clean |
| E2_Roger_Williams__Laras_Theme.mid | Lara's Theme | Hot 100 | clean |
| E2_The_Shadows__Cavatina.mid | Cavatina | UK | clean |
| E2_The_Shadows__Walk_Dont_Run.mid | Walk Don't Run（Shadows） | カバー | clean |
| E2_The_Ventures__Apache.mid | Apache（Ventures） | カバー | clean |
| E2_The_Ventures__Diamond_Head.mid | Diamond Head | 弱 | clean |
| E2_Herb_Alpert__Tijuana_Taxi.mid | Tijuana Taxi | Hot 100 | clean |
| E2_Herb_Alpert__A_Banda.mid | A Banda | Alpert 期 | clean |
| E2_Herb_Alpert__South_of_the_Border.mid | South of the Border | Alpert | clean |
| E2_The_Shadows__Riders_in_the_Sky.mid | Riders in the Sky | Shadows | clean |
| E2_The_Shadows__Guitar_Boogie.mid | Guitar Boogie | 弱 | clean |
| E2_The_Ventures__Blue_Star.mid | Blue Star | Ventures | clean |
| E2_The_Ventures__Honky_Tonk.mid | Honky Tonk | カバー | clean |
| E2_The_Ventures__Bulldog.mid | Bulldog（Ventures） | カバー | clean |

### 4-3. 長尺差し替えで追加（2026-11-21）

| ファイル | 曲 | 注 | 取得 |
|---|---|---|---|
| Dick_Dale__Misirlou.mid | Dick Dale – Misirlou | サーフ定番 | bitmidi |
| Floyd_Cramer__Root_Beer_Rag.mid | Floyd Cramer – Root Beer Rag | Cramer インスト | clean |
| The_Shadows__Quartermaster_Stores.mid | The Shadows – Quartermaster Stores | Shadows インスト | clean |
| The_Ventures__The_Cruel_Sea.mid | The Cruel Sea（Ventures） | Dakotas チャート曲のカバー | clean |
| The_Ventures__Instant_Guitar.mid | Instant Guitar | Ventures（弱） | clean |
| The_Ventures__Escape.mid | Escape | Ventures（弱） | clean |
| The_Ventures__Jokers_Wild.mid | Jokers Wild | Ventures（弱） | clean |
| The_Ventures__Fugitive.mid | Fugitive | Ventures（弱） | clean |
| The_Ventures__Bird_Rockers.mid | Bird Rockers | Ventures（弱） | clean |
| The_Ventures__Eleventh_Hour.mid | Eleventh Hour | Ventures（弱） | clean |
| The_Ventures__Theme_From_Come_September.mid | Theme from Come September | Ventures | clean |
| The_Ventures__Gringo.mid | Gringo | Ventures（弱） | clean |

---

## 5. 基準の強さ

| 強さ | 意味 |
|---|---|
| 強 | チャート表に曲名＋アーティスト一致 |
| 中 | チャート曲のカバー／別名義 |
| 弱 | 定番だがピーク薄い、または差し替え用の短尺 Ventures 等 |

---

## 6. やらなかったこと

- LMD-full の一括ダウンロード  
- アーティストフォルダ丸ごとダンプ  
- 自動スコアだけで採用決定  
- 商用音源の WAV 直集め（MIDI→同一 SF）  
- 長さ合わせのための先頭N秒切り捨て（曲ごと差し替えで対応）  
