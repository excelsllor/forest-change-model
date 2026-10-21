# TypeNet — финальная архитектура (cross-attention readout)

**Финальный чекпоинт:** `typenet_big_w256_b2_attn.pth` (model repo v4.0)
**ONNX (плагин v0.17):** `typenet_encoder.onnx` (30.3 МБ) + `typenet_cfc.onnx` (9.3 МБ, динамическое T)
**Схема:** `docs/11_typenet_final.png`

Задача: для чипа 32×32 px (320 м), где маска нашла изменение, определить тип:
**logging / windfall / fire** (`no_change` не классифицируется — его отсекает маска).

---

## 1. Входы

| вход | форма | смысл |
|---|---|---|
| `vol` (энкодер) | (M,6,32,32) | сырой рефлектанс/10000, clamp 0..1.5 (B02,B03,B04,B08,B11,B12) |
| `feats` (readout) | (M,T,768) | признаки энкодера на дату |
| `dt` (readout) | (M,T) | годы между датами (dt[0]=0) |
| `doy` (readout) | (M,T,3) | (sin DOY, cos DOY, год−2015); для мозаик год=0 |

`T` — **динамическое** (2..8; точные длины без паддинга).

## 2. Энкодер `typenet_encoder.onnx` — CnnDofaLike, 7.58M

| # | слой | операции | форма | параметры |
|---|---|---|---|---|
| 0 | BandConvStem(48,k3,emb64) | w_c = MLP(λ_c) (1→64→64→48·9) → Σ_c conv3×3(x_c,w_c) | (M,48,32,32) | 32.5k |
| 1 | proj | Conv2d 3×3, stride 2, pad 1 + GELU | (M,768,16,16) | 332k |
| 2-13 | 12 × block | dw Conv2d 3×3 (groups=768, dilation d, pad=d) + GroupNorm(8) + GELU + Conv2d 1×1 (768→768) + GroupNorm(8); residual: h = GELU(h + blk(h)) | (M,768,16,16) | 12×1.18M |
| 14 | norm + pool | GroupNorm(8) → AdaptiveAvgPool2d(1) | (M,768) | 1.5k |

- Дилатации блоков: **[1,2,4,8, 1,2,4,8, 1,2,4,8]**; рецептивное поле ≈185 px.
- Скорость (DirectML): ~2.4–2.5 мс/чип-дата.

## 3. Readout `typenet_cfc.onnx` — TypeNetFeats + AttnReadout, 2.33M (динамическое T)

| # | модуль | операции | форма | параметры |
|---|---|---|---|---|
| 0 | token proj | x = [feats 768, doy 3, dt 1] → Linear(772→256) + обучаемое слотовое эмбеддинг-поле (64×256) | (M,T,256) | 198k |
| 1 | запросы | 2 обучаемых query-токена + среднее по валидным датам (masked mean) | (M,2,256) | 0.5k |
| 2 | AttnBlock ×2 | cross-attention: q = W_q LN(query), keys/values = W_kv LN(tokens), 8 голов (dh=32), маска валидных дат (t < L), softmax; residual; FFN(256→1024→256) + LayerNorm | (M,2,256) | 2×0.79M |
| 3 | readout | конкатенация 2 query-токенов | (M,512) | — |
| 4 | сводки | mean_T(feats), last−first(feats) по валидным датам | (M,1536) | — |
| 5 | голова | LayerNorm(2048) → Linear(2048→256) → GELU → Dropout(0.2) → Linear(256→3) | (M,3) | 0.53M |

- CfC-рекуррентности нет: cross-attention над датами; на динамическом T работает без ONNX Loop.
- Пустые/паддинговые даты маскируются (`L = число дат`); при пустом ряду выход нулевой.
- Скорость readout: ~0.2 мс/строка на CPU (T≤8).

## 4. Обучение

- Признаки: замороженный энкодер (дистилляция DOFA-B 7.58M, cos 0.866/0.760), кэш `tn_*.npz`.
- Источники: vozh 6030, big4 45000, california 3313, пары 2079, длинные 2307 (изменённые чипы, 3 класса).
- 60 эпох, bs 32 (по 16 log/wind + fire), lr 2e-4, warmup 200 шагов, cosine, CE с весами классов,
  dropout 0.2; выбор чекпоинта по val macro-F1; AUG ×6 для vozh/pairs/long.

## 5. Метрики

**Все датасеты (val, per-class F1):**

| датасет | n | logging | windfall | fire | macro |
|---|---|---|---|---|---|
| **все** | 9875 | **0.947** | **0.938** | **0.973** | **0.911** |
| vozh | 886 | 0.949 | 0.693 | — (n=1) | 0.547 |
| big4 (DISFOR) | 7200 | 0.859 | 0.910 | 0.935 | 0.901 |
| california | 1471 | — | — | 0.987 | 0.987 |
| pairs (DISFOR) | 266 | 0.964 | 0.945 | 1.000 | 0.970 |
| long (DISFOR) | 52 | 0.900 | 0.857 | 1.000 | 0.919 |

**Vozh по всем классам** (val, n=886, accuracy 0.912):

| класс | n | precision | recall | F1 |
|---|---|---|---|---|
| logging | 746 | 0.934 | 0.965 | 0.949 |
| windfall | 139 | 0.765 | 0.633 | 0.693 |
| fire | 1 | 0.000 | 0.000 | 0.000 |

- GT-чипы p3 через пайплайн плагина: accuracy 0.799; logging F1 0.875 (recall 0.791),
  windfall F1 0.824 (recall 0.821), fire 0.750 (n=4).
- Слабое место: vozh windfall (0.693, 139 val / 291 train).
- readout на CPU; энкодер DML ~2.4 мс/чип-дата; типы p3 — **~140 с cold** (кэш фич в `out_dir/cache`).

## 6. Файлы

- `src/typenet_v3.py` — CnnDofaLike; `src/typenet_big.py` — TypeNetFeats + **AttnReadout**.
- `src/cfc.py` — CfCCell/CfCBiStack (используется маской).
- `src/train_typenet_big.py` — тренер (`--readout attn`); `src/extract_typenet_feats.py` — кэш фич;
  `src/eval_typenet_big.py` — оценка; `src/export_onnx_new.py` — экспорт для плагина.
