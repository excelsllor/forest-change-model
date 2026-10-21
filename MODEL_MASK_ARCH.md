# Модель МАСКИ изменения — финальная архитектура (cross-attention контекст)

**Финальный чекпоинт:** `mask_ctx_attn3_len.pth` (model repo v4.0)
**ONNX (плагин v0.17):** `masknet_encoder.onnx` (5.5 МБ) + `masknet_cfc.onnx` (4.7 МБ, динамическое T)
**Схема:** `docs/10_mask_final_bicfcpn_big.png`

Пайплайн плагина: снимки (6 каналов) → энкодер по каждому чипу/дате → признаки 256-d →
**контекст 3×3 через cross-attention** → CfC (динамическое T) → бинарная маска изменения.

---

## 1. Входы

| вход | форма | смысл |
|---|---|---|
| `vol` (энкодер) | (M,6,32,32) | сырой рефлектанс/10000, clamp 0..1.5; 6 каналов B02,B03,B04,B08,B11,B12 |
| `feats` (CfC) | (M,T,256) | выход энкодера на каждую дату |
| `ctx` (CfC) | (M,T,3,3,256) | признаки 3×3 соседних чипов (тот же энкодер) |
| `valid` (CfC) | (M,T,3,3) | валидность соседних чипов (пустые/нодата = 0) |
| `dt` (CfC) | (M,T) | годы между соседними датами, dt[0]=0 |
| `doy` (CfC) | (M,T,3) | (sin DOY, cos DOY, год−2015); для мозаик год=0 |
| `lengths` (CfC) | (M,) | число валидных дат в ряду (length-aware; T динамическое) |

## 2. Энкодер `masknet_encoder.onnx` — CnnEncoder, 1.41M

| # | слой | операции | форма выхода | параметры |
|---|---|---|---|---|
| 0 | BandConvStem(48,k3,emb64) | w_c = MLP(λ_c): Linear(1→64)+GELU+Linear(64→64)+GELU+Linear(64→432); y = Σ_c conv2d(x_c, w_c, 3×3, pad 1) | (M,48,32,32) | 32.5k |
| 1 | ConvBlock 48→96 | Conv2d 3×3, stride 2, pad 1 + GroupNorm(8) + GELU; skip Conv2d 1×1, stride 2; сумма | (M,96,16,16) | 46.5k |
| 2 | ConvBlock 96→192 | то же | (M,192,8,8) | 185.1k |
| 3 | ConvBlock 192→256 | то же | (M,256,4,4) | 492.5k |
| 4 | ConvBlock 256→256 | то же | (M,256,2,2) | 656.4k |
| 5 | AdaptiveAvgPool2d(1) | GAP | (M,256) | — |

- Все ConvBlock — dilation=1 (дилатация на stride-2 свёртках даёт 220× замедление на MIOpen).
- Скорость (DirectML): 0.036 мс/чип при батче 512.

## 3. CfC `masknet_cfc.onnx` — FeatMaskNet + CtxAttn, 1.16M (динамическое T)

| # | модуль | операции | форма | параметры |
|---|---|---|---|---|
| 0 | **CtxAttn** | query = центр (LayerNorm → Linear 256→256, 8 голов); keys/values = 9 соседей (LayerNorm → Linear 256→512) + обучаемое соседнее эмбеддинг-поле (9×256); scores = q·kᵀ/√32, маска невалидных соседей, softmax; out = W_o(attn·v); residual + FFN(256→512→256) + LayerNorm → 256 | (M,T,256) | ~530k |
| 1 | concat | [feats 256, ctx_vec 256, doy 3, dt 1] | (M,T,516) | — |
| 2 | pre-norm + in_proj | LayerNorm(516) → Linear(516→128) | (M,T,128) | 66.2k |
| 3 | CfC_fwd / CfC_bwd | CfCCell(h=128, backbone 128×2): z = [x_t, h] → Linear(256→128)+LeakyReLU(0.1)+Linear(128→128)+LeakyReLU; gate = σ(time_a(z)·dt_norm + time_b(z)); h' = (1−gate)·ff1(z) + gate·ff2(z); residual y = xp + h | два прохода по T | 2×105k |
| 4 | length-aware пулинг | **среднее по валидным шагам** (s < lengths); обратный проход стартует с последней валидной даты; out = [mean(y_f), mean(y_b)] | (M,256) | — |
| 5 | сводки | mean_T(feats), last−first(feats) по валидным датам; mean_T(ctx_vec), last−first(ctx_vec) | (M,1024) | — |
| 6 | голова | LayerNorm(1280) → Linear(1280→256) → GELU → Dropout(0.2) → Linear(256→2), bias [1,−1] | (M,2) | ~0.37M |

- Если окрестность пуста (ctx-drop/нодата), attention-выход обнуляется — модель работает как без контекста.
- `dt_norm` = dt / средний положительный интервал ряда.
- T динамическое (2..8), `lengths`; CfC-часть исполняется на CPU (ONNX Loop на DirectML некорректен).
- В плагине даты маски ограничены 4 (диапазон обучения 2–4).

## 4. Обучение

- Данные: Vozh — 6030 чипов (p1/p2/p3 = 2/3/4 даты), сплит по кварталам; контекст — из мозаик.
- Переменное T: батч паддится до max длины, `lengths` передаются; length-aware пулинг.
- Энкодер заморожен (дистилляция DOFA-B), обучаются CtxAttn + CfC + голова.
- 60 эпох, bs 32, lr 1.5e-4, AdamW, cosine; ctx-drop 0.3, label smoothing 0.05;
  стоп только при переобучении (train−val > 0.08 два замера подряд), не раньше 30-й эпохи.

## 5. Метрики

**Vozh (val, точные длины 2–4, с контекстом):**

| метрика | значение |
|---|---|
| F1 / P / R | **0.929 / 0.951 / 0.907** |
| p1 / p2 / p3 | 0.922 / 0.959 / 0.917 |
| clean-val (окна без пересечения кварталов) | 0.947 |
| GT p3 через плагин (labeled chips) | **P 0.982 / R 0.967 / F1 0.975** (p3 val-подмножество 0.917) |
| zero-vol F1 | 0.008 |

**Все датасеты (val):**

| датасет | n | pos | P | R | F1 | примечание |
|---|---|---|---|---|---|---|
| vozh (с контекстом) | 1321 | 886 | 0.951 | 0.907 | **0.929** | целевой домен |
| vozh (контекст пуст) | 1321 | 886 | 0.944 | 0.887 | 0.914 | fallback |
| big4 (DISFOR pairs) | 9600 | 9600 | 1.000 | 0.211 | 0.348 | zero-shot |
| california | 3973 | 1471 | 0.000 | 0.000 | 0.000 | zero-shot |
| pairs (DISFOR) | 439 | 266 | 0.582 | 0.241 | 0.340 | zero-shot |
| long (DISFOR, T≤8) | 491 | 335 | 0.656 | 0.188 | 0.292 | zero-shot |

## 6. Скорость

| шаг | значение |
|---|---|
| энкодер (DML) | 0.036 мс/чип (батч 512) |
| CtxAttn (в составе CfC, CPU) | ~0.01 мс/чип-дата |
| маска p3, полная сцена | **42–65 с** |
| ONNX | 5.5 + 4.7 МБ |

## 7. Файлы

- `src/masknet_v2.py` — CnnEncoder, ConvBlock, BandConvStem; `src/cfc.py` — CfCCell/CfCBiEncoderPN.
- `src/exp_ctx_mask.py` — FeatMaskNet + **CtxAttn** (cross-attention), тренер.
- `src/build_vozh_ctx.py` — контекстный датасет; `src/onnx_opt.py` — фолд λ-стема.
- `src/export_onnx_mask_ctx.py` — экспорт (`--ctx attn`).
- Датасет: `outputs/vozh/mask_eval/vozh_ctx_mask.npz`.
