# Forest Change — trained models (final)

Финальные модели задачи выявления и классификации лесных изменений (вырубка / ветровал /
пожар) по Sentinel-2 для QGIS-плагина [`forest-change-qgis`](https://github.com/excelsllor/forest-change-qgis)
(v0.15+). Веса лежат в git (все файлы < 100 МБ), ONNX-экспорты — в git плагина.

Два шага: **маска изменения** (бинарно) → **тип** (logging / windfall / fire).

| модель | чекпоинт | размер | ONNX (плагин) |
|---|---|---|---|
| Маска (контекст 3×3 + length-aware CfC) | `mask_ctx_conv3_len.pth` | 4.7 МБ | `masknet_encoder.onnx` (5.5 МБ) + `masknet_cfc.onnx` (4.9 МБ) |
| TypeNet (DOFA-дистиллят 12×768 + CfC 2×256) | `typenet_big_w256_b2.pth` | 8.4 МБ | `typenet_encoder.onnx` (30.3 МБ) + `typenet_cfc.onnx` (10.9 МБ) |
| Энкодеры (дистилляция DOFA-B) | `encoder_dofa_distilled.pth` (5.4 МБ), `encoder_dofa_distilled_big.pth` (29 МБ) | | |

Подробные послойные описания и схемы: **[MODEL_MASK_ARCH.md](MODEL_MASK_ARCH.md)**,
**[MODEL_TYPES_ARCH.md](MODEL_TYPES_ARCH.md)**, диаграммы в [docs/](docs/).

---

## 1. Маска: `mask_ctx_conv3_len.pth`

**Входы:** `vol` (B,T,6,32,32) — сырой рефлектанс/10000 (clip 0..1.5, без z-норм);
`dt` (B,T) — годы между датами; `doy` (B,T,3) — (sin DOY, cos DOY, год−2015; для мозаик год=0);
`lengths` (B,) — число валидных дат; **T динамическое** (2..8).
**Архитектура:** CnnEncoder (1.41M: λ-стем, свёрнутый при экспорте в 1 dense-Conv(6→48), +
4 ConvBlock 48→96→192→256→256, GAP → 256-d/дата) + контекст 3×3 (Conv3×3 по фичам соседей,
маска валидности) + двунаправленный CfC (128, length-aware пулинг: среднее по валидным шагам,
обратный проход с последней валидной даты) + сводки mean/last−first + голова 1280→256→2.
**Обучение:** Vozh 6030 чипов, сплит по кварталам, энкодер заморожен; 60 эпох, lr 1.5e-4,
ctx-drop 0.3; ранний стоп только при явном переобучении (после 30-й эпохи).

Метрики на всех датасетах (val, бинарно; P/R/F1):

| датасет | n | pos | P | R | F1 | примечание |
|---|---|---|---|---|---|---|
| **vozh** (с контекстом) | 1321 | 886 | 0.938 | 0.924 | **0.931** | целевой домен |
| vozh (пустой контекст) | 1321 | 886 | 0.938 | 0.894 | 0.916 | fallback-режим |
| big4 (DISFOR pairs) | 9600 | 9600 | 1.000 | 0.166 | 0.284 | zero-shot |
| california | 3973 | 1471 | 1.000 | 0.001 | 0.001 | zero-shot |
| pairs (DISFOR) | 439 | 266 | 0.632 | 0.207 | 0.312 | zero-shot |
| long (DISFOR, T≤8) | 491 | 335 | 0.619 | 0.116 | 0.196 | zero-shot |

- Vozh по периодам: p1 0.925 / p2 0.960 / p3 0.906; clean-val (без пересечения окон кварталов) **0.940**;
  GT p3 на сцене через плагин: **P 0.966 / R 0.952 / F1 0.959**.
- Модель обучена на Vozh; zero-shot на DISFOR/California без контекста слабый — для других
  регионов нужно дообучение.
- Скорость: энкодер 0.036 мс/чип (DML, батч 512); вся сцена p3 (98 тайлов) — **59–65 с**
  (энкодер DML + CfC на CPU).

## 2. Тип: `typenet_big_w256_b2.pth`

**Входы:** `feats` (M,T,768) — замороженный дистиллированный энкодер CnnDofaLike (λ-стем +
12 depthwise-separable слоёв ×768, дилатации [1,2,4,8]×3, RF≈185); `dt` (M,T) — годы;
`doy` (M,T,3). **T динамическое** (2..8, ONNX Loop, точные длины).
**Архитектура:** CfCBiStack ×2 (ширина 256, двунаправленный, mean по T) + сводки
mean/last−first (1536) → голова 2048→256→3. CfC исполняется на CPU (Loop на DirectML врёт).
**Обучение:** vozh 6030 + big4 45000 + california 3313 + пары 2079 + длинные 2307 (изменённые чипы,
3 класса, без no_change); 60 эпох, bs 32, lr 2e-4, CE с весами классов, AUG ×6 для vozh/pairs/long.

Метрики (val, per-class F1):

| датасет | n | logging | windfall | fire | macro |
|---|---|---|---|---|---|
| **все** | 9875 | **0.922** | **0.877** | **0.983** | **0.870** |
| vozh | 886 | 0.944 | 0.698 | — (n=1) | 0.547 |
| big4 (DISFOR) | 7200 | 0.816 | 0.830 | 0.934 | 0.860 |
| california | 1471 | — | — | 0.972 | 0.972 |
| pairs (DISFOR) | 266 | 0.921 | 0.879 | 0.959 | 0.920 |
| long (DISFOR) | 52 | 0.927 | 0.889 | 1.000 | 0.939 |

- GT-чипы p3 через ONNX-пайплайн плагина: recall logging 88% / windfall 94%.
- Слабое место: vozh windfall (0.698).
- Скорость: энкодер ~2.4 мс/чип-дата (DML); типы p3 (13 385 клеток) — **177 с cold / 43 с warm**
  (дисковый кэш фич, ключ = даты + энкодер + mtime данных).

## 3. Код (`code/`)

| файл | что это |
|---|---|
| `masknet_v2.py` | CnnEncoder (маска), BandConvStem, ConvBlock, MaskNetV2 |
| `cfc.py` | CfCCell, CfCBiEncoderPN (length-aware), CfCBiStack |
| `typenet_v3.py`, `typenet_big.py` | CnnDofaLike (энкодер типов), TypeNetFeats (CfC-голова) |
| `exp_ctx_mask.py` | FeatMaskNet (контекст 3×3) + тренер маски |
| `build_vozh_ctx.py` | контекстный датасет маски (сетка фич малого энкодера) |
| `train_big.py`, `distill_encoder.py` | обучение маски и дистилляция энкодера |
| `train_typenet_big.py`, `extract_typenet_feats.py`, `eval_typenet_big.py` | TypeNet: тренер, кэш фич, оценка |
| `eval_mask_all.py` | оценка маски на всех датасетах |
| `onnx_opt.py` | фолд λ-стема (6 свёрток → одна dense-Conv) |
| `export_onnx_mask_ctx.py`, `export_onnx_new.py` | экспорт ONNX (маска; типы с динамическим T) |
| `draw_arch.py` | блок-схемы архитектур |

**Воспроизведение:**
```bash
python code/build_vozh_ctx.py --with-big-center       # контекстные датасеты
python code/exp_ctx_mask.py --ctx conv --k 3          # маска (length-aware)
python code/train_typenet_big.py --width 256 --blocks 2 \
    --head-width 256 --head-mode summary              # TypeNet
python code/export_onnx_mask_ctx.py                   # ONNX маски (динамическое T)
python code/export_onnx_new.py                        # ONNX типов (динамическое T)
```

**Данные:** Vozh — `dataset.csv` + мозаики `outputs/vozh/mosaic`; DISFOR — `G:/datasets/unified/big4_chips.npz`;
California — `california.npz`; длинные ряды — `binary_long.npz`; пары — `binary_ds.npz`;
кэши фич типов — `tn_{vozh,big4,cal,pairs,long}.npz`.

## Примечания

- Старые модели (MaskNet-CK, TypeNet-CK, production 3D-CNN+CfC, fusion DOFA+CfC, промежуточные
  варианты) удалены из репозитория.
- Воспроизводимость: детерминированные сиды (0), скрипты в `code/`, конфиги — в CLI-аргументах.
