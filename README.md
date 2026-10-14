# Forest Change — trained models

Веса обученных моделей для задачи выявления и классификации лесных изменений
(вырубка / ветровал / пожар) по Sentinel-2. Используются QGIS-плагином
`forest-change-qgis` (модель в ONNX) и пайплайном обучения.

Файлы весов выложены в **Releases** этого репозитория (в git они не входят:
263 МБ и 124 МБ превышают лимит GitHub в 100 МБ на файл).

## Ассеты релиза

| файл | размер | что это | куда класть |
|---|---|---|---|
| `forest_change_mt_256.onnx` | ~263 МБ | ChangeMamba-Tiny (multi-task: изменение + тип), tile 256, 3 канала B11/B08/B04 | `forest_change/model/` в QGIS-плагине |
| `space_swir_sqrt_last.pth` | ~125 МБ | Основная 4-классовая ChangeMamba (spatial-протокол, SWIR-композит, sqrt-веса классов) | пайплайн обучения / инференс `vozh_infer.py` |
| `mt_over2_fold0.pth` | ~125 МБ | Multi-task ChangeMamba с усиленным oversampling ветровала/пожара (лучшая по ветровалу) | пайплайн обучения |

## Параметры ONNX-модели

- вход: 2 тензора `(1, 3, 256, 256)`, каналы **B11, B08, B04**;
- нормализация: `x/10000` → `clip(x/0.35, 0, 1) × 255` → ImageNet mean/std
  (mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`);
- выход: `concat(change[2], type[3])` — вероятностные логиты бинарного изменения
  и типа нарушения;
- классы: `0 no_change, 1 logging, 2 windfall, 3 fire`;
- сверка PyTorch ↔ onnxruntime: max|Δ| = 7.6e-06.

## Как использовать

QGIS-плагин (CPU-инференс через `onnxruntime`):

1. скачать `forest_change_mt_256.onnx` из Releases;
2. положить в `forest_change/model/` (рядом с `forest_change_mt_256.json`);
3. `pip install onnxruntime` в Python QGIS;
4. запустить плагин и указать путь к модели.

## Метрики (для справки)

Patch-level, пространственный 3-fold CV на наборе Вожегодского лесничества
(Вологодская обл.), held-out кварталы:

| класс | F1 |
|---|---|
| no_change | 0.989 ± 0.004 |
| logging | **0.934 ± 0.002** |
| windfall | 0.620 ± 0.034 (multi-task + oversample: 0.755 ± 0.020) |
| fire | 0.827 ± 0.058 |

Полноразмерная карта (held-out кварталы): logging F1 0.932, IoU 0.873.

## Обучение (кратко)

ChangeMamba (VMamba-Tiny) + 4-классовая голова / multi-task голова;
чисто-PyTorch реализация selective scan (запуск без CUDA-ядер, в т.ч. на ROCm);
CfC (liquid neural network) — как отдельная временная модель и для
рекуррентной глубины.
