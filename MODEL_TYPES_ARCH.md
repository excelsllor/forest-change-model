# TypeNet — финальная архитектура (классификация типа изменения)

**Финальный чекпоинт:** `typenet_big_w256_b2.pth` (model repo v2.5)
**ONNX (плагин v0.15+):** `typenet_encoder.onnx` (30.3 МБ) + `typenet_cfc.onnx` (10.9 МБ, динамическое T)
**Схема:** `outputs/arch/11_typenet_final.png` (PNG + SVG)

Задача: для чипа 32×32 px (320 м), где маска нашла изменение, определить тип:
**logging / windfall / fire** (`no_change` не классифицируется — его отсекает маска).

---

## 1. Входы

| вход | форма | смысл |
|---|---|---|
| `vol` (энкодер) | (M,6,32,32) | сырой рефлектанс/10000, clamp 0..1.5 (B02,B03,B04,B08,B11,B12) |
| `feats` (CfC) | (M,T,768) | признаки энкодера на дату |
| `dt` (CfC) | (M,T) | годы между датами (dt[0]=0) |
| `doy` (CfC) | (M,T,3) | (sin DOY, cos DOY, год−2015); для мозаик год=0 |

`T` — **динамическое** (2..8; ONNX Loop, точные длины без паддинга).

## 2. Энкодер `typenet_encoder.onnx` — CnnDofaLike, 7.58M

| # | слой | операции | форма | параметры |
|---|---|---|---|---|
| 0 | BandConvStem(48,k3,emb64) | w_c = MLP(λ_c) (1→64→64→48·9) → Σ_c conv3×3(x_c,w_c); **при экспорте свёрнут в одну dense-Conv(6→48)** | (M,48,32,32) | 32.5k |
| 1 | proj | Conv2d 3×3, stride 2, pad 1 + GELU | (M,768,16,16) | 332k |
| 2-13 | 12 × block | dw Conv2d 3×3 (groups=768, dilation d, pad=d) + GroupNorm(8) + GELU + Conv2d 1×1 (768→768) + GroupNorm(8); residual: h = GELU(h + blk(h)) | (M,768,16,16) | 12×1.18M |
| 14 | norm + pool | GroupNorm(8) → AdaptiveAvgPool2d(1) | (M,768) | 1.5k |

- Дилатации блоков: **[1,2,4,8, 1,2,4,8, 1,2,4,8]**; рецептивное поле ≈185 px (покрывает чип).
- Пространственное разрешение не уменьшается (16×16 на всех 12 блоках).
- Скорость (DirectML): ~2.4–2.5 мс/чип-дата.

## 3. CfC `typenet_cfc.onnx` — TypeNetFeats, 2.70M (динамическое T, ONNX Loop)

| # | модуль | операции | форма | параметры |
|---|---|---|---|---|
| 0 | concat | [feats 768, doy 3, dt 1] | (M,T,772) | — |
| 1 | CfCBiBlock 1 | LayerNorm(772) → Linear(772→256); CfC_fwd/CfC_bwd (CfCCell h=256, backbone 256×2): z = [x,h] → Linear(512→256)+LeakyReLU(0.1)+Linear(256→256)+LeakyReLU; gate = σ(time_a(z)·dt_norm + time_b(z)); h' = (1−gate)·ff1(z) + gate·ff2(z); y = concat(y_f, y_b) = 512 (per-step, residual) | (M,T,512) | 1.19M |
| 2 | CfCBiBlock 2 | LayerNorm(512) → Linear(512→256) → то же | (M,T,512) | 0.99M |
| 3 | mean по T | — | (M,512) | — |
| 4 | сводки | mean_T(feats), last−first(feats) по валидным датам | (M,1536) | — |
| 5 | голова | LayerNorm(2048) → Linear(2048→256) → GELU → Dropout(0.2) → Linear(256→3) | (M,3) | 0.53M |

- `dt_norm` = dt / средний положительный интервал ряда.
- Экспорт: TorchScript + **ONNX Loop** (`T` динамическое); проверено T=2..8 (max|diff| ~1e-6).
- CfC на **CPU** (Loop на DirectML даёт неверный результат): 0.067 мс/строка (T=4), 0.147 (T=8).

## 4. Обучение

- Признаки: замороженный энкодер (дистилляция DOFA-B 7.58M, cos 0.866/0.760), кэш `tn_{vozh,big4,cal,pairs,long}.npz`.
- Источники (изменённые чипы): vozh (6030, T≤4), big4 DISFOR (45000, T=2), california (3313, T=2),
  пары DISFOR (2079, T=2), длинные ряды (2307, T≤36, подвыборка ≤8 при обучении).
- 60 эпох, bs 32 (по 16 log/wind + fire), lr 2e-4, warmup 200 шагов, cosine, CE с весами классов,
  dropout 0.2; выбор чекпоинта по val macro-F1 (3 класса); AUG ×6 (повороты/флипы) для vozh/pairs/long.
- Голова: широкая (256) + **сводки mean/last−first** — ключевое изменение, давшее +0.033 macro.

## 5. Метрики на всех датасетах (val, per-class F1)

| датасет | n | logging | windfall | fire | macro |
|---|---|---|---|---|---|
| **все** | 9875 | **0.922** | **0.877** | **0.983** | **0.870** |
| vozh | 886 | 0.944 | 0.698 | — (n=1) | 0.547 |
| big4 (DISFOR) | 7200 | 0.816 | 0.830 | 0.934 | 0.860 |
| california | 1471 | — | — | 0.972 | 0.972 |
| pairs (DISFOR) | 266 | 0.921 | 0.879 | 0.959 | 0.920 |
| long (DISFOR) | 52 | 0.927 | 0.889 | 1.000 | 0.939 |

- macro по присутствующим классам (у california только fire; у vozh 1 fire-сэмпл в val).
- GT-чипы p3 через ONNX-пайплайн плагина: recall logging 88% / windfall 94%.
- Слабое место: vozh windfall (0.698) — редкий класс (291 train-сэмпл против 9600 big4).

## 6. Скорость

| шаг | значение |
|---|---|
| энкодер (DML) | ~2.4 мс/чип-дата |
| CfC (CPU, динамическое T) | 0.067/0.147 мс/строка (T=4/8) |
| типы p3 (13 385 клеток) | **177 с cold / 43 с warm** (дисковый кэш фич, ключ = даты + энкодер + mtime данных) |
| ONNX | 30.3 + 10.9 МБ |

## 7. Файлы

- `src/typenet_v3.py` — CnnDofaLike; `src/cfc.py` — CfCCell/CfCBiStack; `src/typenet_big.py` — TypeNetFeats;
  `src/train_typenet_big.py` — тренер; `src/extract_typenet_feats.py` — кэш фич; `src/eval_typenet_big.py`;
  `src/export_onnx_new.py` — экспорт (энкодер + динамический CfC); `src/onnx_opt.py` — фолд λ-стема.
