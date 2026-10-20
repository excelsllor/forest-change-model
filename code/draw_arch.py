"""Graphical block diagrams of the model architectures (PNG + SVG).

Outputs to outputs/arch/:
  01_pipeline.png          - full plugin pipeline (mask model -> type model -> outputs)
  02_mask_3dcnn_cfc.png    - production mask model (3D-CNN + CfC)
  03_masknet_ck.png        - MaskNet-CK (channel-agnostic + CKConv + CfC)
  04_type_fusion.png       - production type model (siamese DOFA + CfC + logit-stack)
  05_typenet_ck.png        - TypeNet-CK (DOFA features + CKConv + CfC)
  06_masknet_v2.png        - MaskNet-v2 (ViT encoder experiments)
  07_typenet_v3.png        - TypeNet-v3 (DOFA/CNN encoder + temporal vector + 2x CfC) [current type]
  08_masknet_v2_cnn.png    - MaskNet-v2 CNN + CfC [current mask]
  09_masknet_v2_bicfcpn.png - MaskNet-v2 bicfcpn (pre-norm + residual + bidirectional CfC) [chosen mask]
  10_mask_final_bicfcpn_big.png - ФИНАЛЬНАЯ маска: bicfcpn_big, сырой рефлектанс, замороженный дистиллированный энкодер
  11_typenet_final.png     - ФИНАЛЬНЫЙ TypeNet: дистиллированный энкодер 12x768 + CfC 2x256 (3 класса)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = r"C:\Users\artyom\Desktop\лесной хакатон\outputs\arch"
os.makedirs(OUT, exist_ok=True)

C_IN = "#dbeafe"      # inputs
C_NET = "#dcfce7"     # network blocks
C_FUS = "#fef9c3"     # fusion
C_OUT = "#fce7f3"     # outputs
C_TXT = "#111827"


def new_fig(w=13, h=9, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    return fig, ax


def box(ax, x, y, w, h, text, color=C_NET, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.2, edgecolor="#374151", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C_TXT,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, color="#374151", style="-|>", lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[arch] saved {name}.png/.svg", flush=True)


# ---------------------------------------------------------------- 1) pipeline
def pipeline():
    fig, ax = new_fig(14, 8, "Пайплайн плагина: маска (модель) → тип (модель) → выходы")
    box(ax, 2, 78, 22, 12, "Снимки Sentinel-2\n(2–36 дат, 6 каналов,\n32 px чипы = 320 м)", C_IN, 9)
    box(ax, 28, 82, 26, 12, "МАСКА: 3D-CNN + CfC\nvol (T,6,32,32) + seq (T,9) + dt (T)\n"
        "бинарно: изменение / нет\nF1 0.943", C_NET, 8.5, bold=True)
    box(ax, 28, 62, 26, 12, "Нелесной фильтр\nSCL (город/вода) + NDVI<0.3\n→ 255", C_FUS, 8.5)
    box(ax, 58, 78, 26, 12, "ТИП: fusion DOFA + CfC\npre/post 128px + seq (4,9)\n"
        "4 класса: no_change / logging /\nwindfall / fire, macro-F1 0.938", C_NET, 8.5, bold=True)
    box(ax, 58, 58, 26, 10, "Только изменённые чипы\n(маска отфильтровала\nостальное)", C_FUS, 8.5)
    box(ax, 20, 40, 24, 11, "classes.tif\n(0/1/2/3, 255 nodata)", C_OUT, 9)
    box(ax, 50, 40, 22, 11, "cells.gpkg\n320 м: координаты,\nтип, уверенность, даты", C_OUT, 9)
    box(ax, 76, 40, 22, 11, "changes.gpkg\nmerged polygons", C_OUT, 9)
    box(ax, 20, 22, 78, 10, "QGIS: маска красным на прозрачном, классы по типам "
        "(logging красный, windfall золотой, fire маджента), Identify по клетке", C_FUS, 9)
    arrow(ax, (24, 84), (28, 88)); arrow(ax, (41, 82), (41, 74))
    arrow(ax, (54, 88), (58, 84)); arrow(ax, (71, 78), (71, 68))
    arrow(ax, (54, 68), (58, 63))
    arrow(ax, (13, 90), (71, 90), rad=0.25, color="#6b7280", ls="--")
    arrow(ax, (32, 62), (32, 51)); arrow(ax, (58, 60), (44, 51), rad=0.1)
    arrow(ax, (58, 63), (58, 51)); arrow(ax, (76, 58), (82, 51))
    arrow(ax, (60, 40), (60, 32))
    save(fig, "01_pipeline")


# ---------------------------------------------------------------- 2) mask model
def mask_3dcnn_cfc():
    fig, ax = new_fig(14, 10, "Модель МАСКИ (production): 3D-CNN + CfC, 139k параметров, F1 0.943")
    box(ax, 4, 86, 26, 10, "vol (B,T,6,32,32)\nобъём снимков, per-scene\nнормализация (med/MAD)", C_IN)
    box(ax, 37, 86, 26, 10, "seq (B,T,9)\n6 каналов + NDVI/NBR/NDMI\n(пулинг по чипу)", C_IN)
    box(ax, 70, 86, 26, 10, "dt (B,T)\nгоды между датами\n(dt[0]=0)", C_IN)
    box(ax, 4, 66, 26, 14, "3D-CNN:\nConv3d(6→16,k3)+GN+GELU\nConv3d(16→32,k3,stride(1,2,2))\n"
        "Conv3d(32→64,k3,stride(1,2,2))\nAdaptiveAvgPool3d(1)", C_NET, 8)
    box(ax, 70, 66, 26, 14, "CfC:\nCfCEncoder(9→96)\nbackbone 96×2, pool=mean\n"
        "sigmoid(time_a·dt + time_b)\n→ рекуррентно по любой T", C_NET, 8)
    box(ax, 4, 54, 26, 6, "64-d (любая T)", C_FUS, 9)
    box(ax, 70, 54, 26, 6, "96-d (любая T, реальные dt)", C_FUS, 9)
    box(ax, 30, 38, 40, 10, "FUSION: concat(64, 96) = 160\nLinear(160→64) + ReLU + Dropout(0.2)\n"
        "Linear(64→2)", C_FUS, 9.5, bold=True)
    box(ax, 38, 22, 24, 8, "logits (B,2)\n0 нет / 1 изменение", C_OUT, 10, bold=True)
    box(ax, 6, 8, 88, 9, "ONNX: T фиксирована (8, добивка нулями) + B динамическая; 0.13–0.31 мс/чип CPU\n"
        "Метрики: F1 0.943 (p1 0.934 / p2 0.967 / p3 0.918 / DISFOR 0.961); T=2/3/4 → 0.940/0.945/0.918",
        C_FUS, 9)
    arrow(ax, (17, 86), (17, 80)); arrow(ax, (83, 86), (83, 80))
    arrow(ax, (50, 86), (83, 80), rad=-0.2, color="#6b7280", ls="--")
    arrow(ax, (17, 66), (17, 60)); arrow(ax, (83, 66), (83, 60))
    arrow(ax, (30, 57), (36, 48), rad=-0.15); arrow(ax, (70, 57), (64, 48), rad=0.15)
    arrow(ax, (50, 38), (50, 30))
    save(fig, "02_mask_3dcnn_cfc")


# ---------------------------------------------------------------- 3) MaskNet-CK
def masknet_ck():
    fig, ax = new_fig(14, 10.5, "MaskNet-CK: любые каналы (генерация весов из λ) + любая T (CKConv + CfC)")
    box(ax, 2, 88, 22, 9, "vol (B,T,C,32,32)\nC каналов", C_IN)
    box(ax, 27, 88, 20, 9, "wl (C)\nдлины волн, мкм", C_IN)
    box(ax, 50, 88, 20, 9, "dt (B,T)", C_IN)
    box(ax, 73, 88, 25, 9, "doy (B,T,3)\nsin/cos DOY + год", C_IN)
    box(ax, 2, 70, 22, 14, "BandProj:\nw_c = MLP_proj(λ_c)\ny = Σ_c w_c · x_c\n→ (B,T,16,32,32)", C_NET, 8)
    box(ax, 27, 70, 20, 14, "Spatial 2D CNN\n(на каждую дату)\nConv 16→32→64→64\n→ (B,T,64)", C_NET, 8)
    box(ax, 50, 70, 20, 14, "BandEmbed:\ne = Σ_c MLP_emb(λ_c)·x̄_c\n→ (B,T,32)", C_NET, 8)
    box(ax, 73, 70, 25, 14, "date_proj:\nconcat(64, 3) → Linear\n→ (B,T,64)", C_NET, 8)
    box(ax, 27, 56, 20, 10, "CKConv-блоки ×2:\nκ(τ)=MLP(τ), τ∈[-1,1]\ny[t]=Σ_τ κ(τ)⊙x[t+τ]\n"
        "(непрерывное ядро)", C_NET, 8)
    box(ax, 73, 56, 25, 10, "concat(32, 3) = 35\n→ CfC(35→96)\nрекуррентный, реальные dt", C_NET, 8)
    box(ax, 27, 44, 20, 6, "mean по T → 64-d", C_FUS, 9)
    box(ax, 73, 44, 25, 6, "96-d", C_FUS, 9)
    box(ax, 32, 28, 36, 10, "FUSION: concat(64, 96) = 160\nMLP(160→64→2)", C_FUS, 9.5, bold=True)
    box(ax, 40, 14, 20, 7, "logits (B,2)", C_OUT, 10, bold=True)
    box(ax, 3, 3, 94, 8, "Метрики: F1 0.769 (p1 0.346 / p2 0.834 / p3 0.835 / DISFOR 0.851); "
        "T=2..36 → 0.62–0.81; каналы: C=6 → 0.769, без B12 → 0.273, [B04,B08,B11,B12] → 0.745",
        C_FUS, 8.5)
    arrow(ax, (13, 88), (13, 84)); arrow(ax, (37, 88), (37, 84)); arrow(ax, (60, 88), (60, 84))
    arrow(ax, (85, 88), (85, 84))
    arrow(ax, (37, 70), (37, 66)); arrow(ax, (85, 70), (85, 66))
    arrow(ax, (37, 56), (37, 50)); arrow(ax, (85, 56), (85, 50))
    arrow(ax, (47, 47), (52, 38), rad=-0.15); arrow(ax, (73, 47), (68, 38), rad=0.15)
    arrow(ax, (50, 28), (50, 21))
    save(fig, "03_masknet_ck")


# ---------------------------------------------------------------- 4) type fusion
def type_fusion():
    fig, ax = new_fig(14, 10, "Модель ТИПА (production): siamese DOFA + CfC + logit-stack, macro-F1 0.938")
    box(ax, 3, 88, 22, 9, "pre (B,6,128,128)\nкрайняя дата окна", C_IN)
    box(ax, 28, 88, 22, 9, "post (B,6,128,128)\nконец окна", C_IN)
    box(ax, 55, 88, 20, 9, "seq (B,4,9)\n4 даты × 9 признаков", C_IN)
    box(ax, 3, 70, 47, 14, "DOFA-B/16 @128px (6 каналов, pos_embed 14×14→8×8, 111.3M):\n"
        "patch_embed(x, λ) — веса генерируются из длин волн (любое число каналов)\n"
        "cls + 8×8 токенов → 12 блоков → mean(токены) → 768-d на дату", C_NET, 8)
    box(ax, 55, 70, 20, 14, "CfC:\nCfCEncoder(9→128)\nbackbone 128×2\npool=mean\n"
        "dt зашит [0,0.72,1.15,1.0]", C_NET, 8)
    box(ax, 3, 58, 47, 7, "concat(feat(pre), feat(post)) = 1536-d → Linear(1536→4) → s_logits",
        C_FUS, 9)
    box(ax, 55, 58, 20, 7, "Linear(128→4)\n→ t_logits", C_FUS, 9)
    box(ax, 28, 40, 40, 10, "FUSION (logit-stack):\nconcat(s_logits(4), t_logits(4)) = 8\n"
        "MLP(8→512→4)", C_FUS, 9.5, bold=True)
    box(ax, 38, 24, 22, 8, "logits (B,4)\nno_change / logging /\nwindfall / fire", C_OUT, 9, bold=True)
    box(ax, 4, 8, 90, 10, "Точность среди изменённых чипов (held-out кварталы): logging 0.942 / windfall 0.966 / "
        "fire 0.907 / macro 0.938\nСкорость: 66.8 мс/чип CPU, 10.1 мс/чип GPU (ROCm, 6.6×); ONNX 426 МБ",
        C_FUS, 9)
    arrow(ax, (14, 88), (14, 84)); arrow(ax, (39, 88), (39, 84)); arrow(ax, (65, 88), (65, 84))
    arrow(ax, (26, 70), (26, 65)); arrow(ax, (65, 70), (65, 65))
    arrow(ax, (26, 58), (40, 50), rad=-0.15); arrow(ax, (65, 58), (56, 50), rad=0.15)
    arrow(ax, (48, 40), (48, 32))
    save(fig, "04_type_fusion")


# ---------------------------------------------------------------- 5) TypeNet-CK
def typenet_ck():
    fig, ax = new_fig(14, 9.5, "TypeNet-CK: DOFA (любые каналы) + CKConv + CfC — типы при любом числе фото")
    box(ax, 2, 88, 28, 9, "DOFA (frozen) на каждую дату окна\n→ (B,T,768)  [любое число каналов]", C_IN)
    box(ax, 34, 88, 28, 9, "spec (B,T,9)\nспектральные признаки", C_IN)
    box(ax, 66, 88, 30, 9, "dt (B,T) + doy (B,T,3)\nвремя и сезон", C_IN)
    box(ax, 6, 66, 40, 14, "CKConv-ветка:\nLinear(768→128)+LN+GELU\n"
        "CKBlock ×2: y = x + CKConv(LN(x))\nκ(τ)=MLP(τ) → любая T\nmean по T → 128-d", C_NET, 8.5)
    box(ax, 54, 66, 40, 14, "CfC-ветка:\n[Linear(768→64), spec 9] = 73 (+ doy)\n"
        "CfCEncoder(73→96), backbone 96×2\nрекуррентный, реальные dt → любая T", C_NET, 8.5)
    box(ax, 30, 48, 40, 10, "FUSION-голова:\nconcat(128, 96) = 224 → MLP(224→64→4)\n"
        "CE + веса классов (клип 20) + label smoothing", C_FUS, 9, bold=True)
    box(ax, 40, 32, 20, 8, "logits (B,4)\nтип изменения", C_OUT, 9.5, bold=True)
    box(ax, 4, 16, 92, 11, "Метрики: среди изменённых Vozh — macro-F1 0.411 (logging 0.798 / windfall 0.436 / fire 0.000)\n"
        "DISFOR 0.541, длинные ряды 0.564; T=2..8 без переобучения (0.26–0.56)\n"
        "Уступает production fusion (0.938): замороженные per-date признаки + mean по T теряют сигнал",
        C_FUS, 8.5)
    box(ax, 4, 2, 92, 10, "Идея: DOFA канало-агностичен (веса из λ), CKConv длина-агностичен (непрерывное ядро),\n"
        "CfC рекуррентный (любая T, реальные dt) → модель работает с любым числом фото и каналов",
        C_NET, 9)
    arrow(ax, (16, 88), (16, 80)); arrow(ax, (44, 88), (66, 80), rad=0.2)
    arrow(ax, (48, 88), (66, 80), rad=0.1); arrow(ax, (81, 88), (81, 80))
    arrow(ax, (26, 66), (40, 58)); arrow(ax, (74, 66), (60, 58))
    arrow(ax, (50, 48), (50, 40))
    save(fig, "05_typenet_ck")


# ---------------------------------------------------------------- 6) MaskNet-v2
def masknet_v2():
    fig, ax = new_fig(14, 10, "MaskNet-v2: DOFA-style энкодер + временной вектор + (CfC | FlexConv | S5) + декодер")
    box(ax, 2, 88, 30, 9, "vol (B,T,C,32,32)\nC любых каналов", C_IN)
    box(ax, 36, 88, 26, 9, "wl (C)\nдлины волн", C_IN)
    box(ax, 66, 88, 32, 9, "dt (B,T) + doy (B,T,3)\nsin/cos DOY, год, Δt", C_IN)
    box(ax, 2, 70, 30, 14, "DOFA-style энкодер (обучаемый, ~2M):\n"
        "patch 4px → 64 токена\nBandPatchEmbed: ядра = MLP(λ_c)\n(любое C) → 4 блока + cls", C_NET, 8.5)
    box(ax, 36, 74, 26, 10, "признак на дату\n(B,T,192)", C_FUS, 9)
    box(ax, 36, 62, 26, 8, "concat с временным\nвектором → (B,T,196)", C_FUS, 8.5)
    box(ax, 2, 44, 30, 14, "A) CfC (hidden 128)\nрекуррентная, реальные dt\n→ 128-d", C_NET, 9)
    box(ax, 35, 44, 30, 14, "B) FlexConv ×2\nκ(τ)=MLP(τ)·exp(−½(τ/σ)²)\nσ — обучаемая полоса, любая T", C_NET, 8.5)
    box(ax, 68, 44, 30, 14, "C) S5 (честный parallel scan)\ndiagonal SSM + Hillis-Steele\nlog₂T тензорных шагов, любая T", C_NET, 8.5)
    box(ax, 30, 28, 40, 9, "ДЕКОДЕР: LayerNorm → Linear(→64) → GELU\n→ Dropout → Linear(64→2), bias инициализ. в «нет»",
        C_FUS, 9, bold=True)
    box(ax, 40, 14, 20, 7, "logits (B,2)\n1 → квадрат закрашен", C_OUT, 9.5, bold=True)
    box(ax, 2, 2, 96, 9, "Точность: CfC 0.829 (P 0.735/R 0.951), S5 0.824, FlexConv 0.772; "
        "T=2..36 стабильно (0.71–0.84), C=4/5/6 → 0.77–0.83\n"
        "Скорость GPU: 0.30 мс/чип (T=2) → 5.0 (T=36); VRAM 82→763 МБ; CPU 37→517 мс/чип (энкодер ×T)",
        C_FUS, 8.5)
    arrow(ax, (17, 88), (17, 84)); arrow(ax, (49, 88), (49, 84)); arrow(ax, (82, 88), (49, 84), rad=0.15)
    arrow(ax, (32, 77), (36, 78)); arrow(ax, (49, 74), (49, 70))
    arrow(ax, (49, 62), (17, 58), rad=0.15); arrow(ax, (49, 62), (50, 58)); arrow(ax, (49, 62), (83, 58), rad=-0.15)
    arrow(ax, (17, 44), (40, 37), rad=-0.1); arrow(ax, (50, 44), (50, 37)); arrow(ax, (83, 44), (60, 37), rad=0.1)
    arrow(ax, (50, 28), (50, 21))
    save(fig, "06_masknet_v2")


# ---------------------------------------------------------------- 7) TypeNet-v3
def typenet_v3():
    fig, ax = new_fig(14, 10.5, "TypeNet-v3 (тип изменения по маске): DOFA/CNN-энкодер + временной вектор + 2×CfC")
    box(ax, 2, 89, 30, 8, "DOFA-B на каждую дату\n(B,T,768) из type_feats.npz\n(frozen, dofa_vozh.pth)", C_IN, 8.5)
    box(ax, 35, 89, 30, 8, "vol (B,T,6,32,32)\nCNN-вариант: end-to-end,\nz-нормализация", C_IN, 8.5)
    box(ax, 68, 89, 30, 8, "spec (B,T,9)\n+ dt (B,T) + doy (B,T,3)\nsin/cos DOY, год, Δt", C_IN, 8.5)
    box(ax, 2, 71, 30, 14, "Вариант A: DOFA-признак\n768-d на дату\n(замороженный ViT-B)", C_NET, 9)
    box(ax, 35, 71, 30, 14, "Вариант B: CnnDofaLike\nλ-стем + 12 depthwise-\nseparable слоёв × 768\n→ GAP → 768-d", C_NET, 9)
    box(ax, 68, 75, 30, 10, "временной вектор на дату:\n(sin DOY, cos DOY, год−2015, Δt)", C_FUS, 8.5)
    box(ax, 15, 57, 70, 9, "concat(768 + 9 + 4) = 781 на дату — «признаки со всех фото + время»", C_FUS, 9.5, bold=True)
    box(ax, 12, 36, 76, 16, "CfC-стек ×2 (ширина 512):\n"
        "CfCCell(h): z = backbone([x, h]); gate = σ(time_a(z)·dt + time_b(z)); h' = (1−gate)·ff1(z) + gate·ff2(z)\n"
        "между слоями LayerNorm + Linear(h→h); mean по T → 512-d (любая длина ряда, реальные dt)", C_NET, 8.5)
    box(ax, 18, 21, 64, 9, "голова: LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.2) → Linear(256→3)",
        C_FUS, 9, bold=True)
    box(ax, 38, 8, 24, 8, "logits (B,3)\nlogging / windfall / fire", C_OUT, 9.5, bold=True)
    box(ax, 2, 0.5, 96, 6.5, "no_change НЕ классифицируется — его отсекает модель маски; обучение только на изменённых чипах "
        "(4150/1432), CE с весами классов, батчи по длине ряда (2/3/4/8)\n"
        "DOFA-512: macro-F1 0.902 среди изменённых (logging 0.974 / windfall 0.882 / fire 0.984), 4.21M; CNN-512: 11.79M, не обучен",
        C_FUS, 8.5)
    arrow(ax, (17, 89), (17, 85)); arrow(ax, (50, 89), (50, 85)); arrow(ax, (83, 89), (83, 85))
    arrow(ax, (17, 71), (40, 66), rad=-0.1); arrow(ax, (50, 71), (50, 66)); arrow(ax, (83, 75), (60, 66), rad=0.1)
    arrow(ax, (50, 57), (50, 52))
    arrow(ax, (50, 36), (50, 30))
    arrow(ax, (50, 21), (50, 16))
    save(fig, "07_typenet_v3")


# ---------------------------------------------------------------- 8) MaskNet-v2 CNN
def masknet_v2_cnn():
    fig, ax = new_fig(14, 10, "MaskNet-v2 CNN + CfC — актуальная модель маски (любая T, любое C)")
    box(ax, 2, 88, 30, 9, "vol (B,T,C,32,32)\nC любых каналов\n(z-нормализация)", C_IN)
    box(ax, 36, 88, 26, 9, "wl (C)\nдлины волн, мкм", C_IN)
    box(ax, 66, 88, 32, 9, "dt (B,T) + doy (B,T,3)\nреальные интервалы + сезон", C_IN)
    box(ax, 2, 66, 60, 18, "CnnEncoder (1.41M), канало-агностичный (ядра из λ):\n"
        "BandConvStem: w_c = MLP(λ_c) → Σ_c conv3×3(x_c, w_c) → 48 каналов\n"
        "ConvBlock ×4 (Conv3×3 stride2 + GroupNorm(8) + GELU + 1×1 stride2 skip):\n"
        "32×32(48) → 16×16(96) → 8×8(192) → 4×4(256) → 2×2(256) → GAP → 256-d", C_NET, 8.5)
    box(ax, 66, 71, 32, 9, "признак на дату\n(B,T,256)", C_FUS, 9)
    box(ax, 66, 58, 32, 9, "concat с временным вектором\n→ (B,T,260)", C_FUS, 8.5)
    box(ax, 16, 40, 68, 13, "CfCEncoder (hidden 128, backbone 128×2, pool=mean):\n"
        "gate = σ(time_a(z)·dt + time_b(z)); h' = (1−gate)·ff1(z) + gate·ff2(z)\n"
        "рекуррентный по датам, реальные dt → любая T → 128-d", C_NET, 9)
    box(ax, 20, 26, 60, 9, "голова: LayerNorm(128) → Linear(128→64) → GELU → Dropout(0.2) → Linear(64→2), bias [1,−1]",
        C_FUS, 8.5, bold=True)
    box(ax, 40, 13, 20, 7, "logits (B,2)\n0 нет / 1 изменение", C_OUT, 9.5, bold=True)
    box(ax, 2, 1, 96, 9.5, "Организаторы (held-out кварталы): F1 0.848 (P 0.896 / R 0.806), p1 0.802 / p2 0.902 / p3 0.840; "
        "смешанный val 0.863\nDISFOR-пары 0.985; длинные ряды 0.800 (T=2..36: 0.76–0.82); fire-пары 317/317 = 1.000; "
        "каналы C=6/5/4 → 0.848/0.351/0.862\nСкорость: GPU 0.12/0.36/1.53 мс/чип (T=2/8/36), CPU 1/4/19; VRAM 749 МБ (T=36); "
        "1.55M параметров; обучение 17.4 мин (16 эпох, bs 32)", C_FUS, 8)
    arrow(ax, (17, 88), (17, 84)); arrow(ax, (49, 88), (49, 84)); arrow(ax, (82, 88), (82, 84))
    arrow(ax, (62, 75), (66, 75))
    arrow(ax, (82, 71), (82, 67))
    arrow(ax, (66, 62), (55, 53), rad=0.1)
    arrow(ax, (50, 40), (50, 35))
    arrow(ax, (50, 26), (50, 20))
    save(fig, "08_masknet_v2_cnn")


# ---------------------------------------------------------------- 9) bicfcpn (chosen mask)
def masknet_v2_bicfcpn():
    fig, ax = new_fig(14, 11, "Выбранная модель маски: MaskNet-v2 bicfcpn — pre-norm + residual + двунаправленный CfC")
    box(ax, 2, 90, 30, 8, "vol (B,T,C,32,32)\nC любых каналов\n(z-нормализация)", C_IN, 8.5)
    box(ax, 36, 90, 26, 8, "wl (C)\nдлины волн, мкм", C_IN, 8.5)
    box(ax, 66, 90, 32, 8, "dt (B,T) + doy (B,T,3)\nΔt и сезон", C_IN, 8.5)
    box(ax, 2, 69, 60, 17, "CnnEncoder (1.41M, без изменений):\n"
        "BandConvStem (λ-ядра) → 48 → ConvBlock ×4 (s2, GN(8), GELU, 1×1 skip):\n"
        "32×32(48) → 16×16(96) → 8×8(192) → 4×4(256) → 2×2(256) → GAP → 256-d", C_NET, 8.5)
    box(ax, 66, 76, 32, 10, "временной вектор:\nsin/cos DOY, год−2015, Δt", C_FUS, 8.5)
    box(ax, 2, 55, 60, 9, "concat(256 + 4) = 260 на дату", C_FUS, 9.5, bold=True)
    box(ax, 2, 40, 30, 11, "pre-norm: u = LayerNorm(260)\n→ xp = Linear(260→128)\n"
        "Δt-нормировка: dt / средний\nинтервал ряда (dt[0]=0)", C_NET, 8.5)
    box(ax, 36, 40, 30, 11, "CfC_fwd(xp, dt_norm)\nи CfC_bwd(xp, dt_norm)\n"
        "реальные dt, любая T\nresidual: y = xp + h", C_NET, 8.5)
    box(ax, 70, 40, 28, 11, "y = mean_T(concat(y_f, y_b))\n= 256-d\n(двунаправленность)", C_FUS, 8.5)
    box(ax, 20, 25, 60, 9, "голова: LayerNorm(256) → Linear(256→64) → GELU → Dropout(0.2) → Linear(64→2), bias [1,−1]",
        C_FUS, 8.5, bold=True)
    box(ax, 40, 11, 20, 7, "logits (B,2)\n0 нет / 1 изменение", C_OUT, 9.5, bold=True)
    box(ax, 2, 0.5, 96, 8, "Организаторы: F1 0.822 (P 0.910 / R 0.749), p1 0.841 / p2 0.857 / p3 0.774; "
        "mixed val 0.844; DISFOR-пары 0.971; длинные 0.790; fire 317/317\n"
        "1.69M параметров; GPU 0.13→1.72 мс/чип (T=2→36), CPU 1→21; обучение 17.3 мин (16 эпох, bs 32)",
        C_FUS, 8.5)
    arrow(ax, (17, 90), (17, 86)); arrow(ax, (49, 90), (49, 86)); arrow(ax, (82, 90), (82, 86))
    arrow(ax, (62, 77), (66, 79))
    arrow(ax, (32, 69), (32, 64)); arrow(ax, (82, 76), (50, 64), rad=0.15)
    arrow(ax, (17, 55), (17, 51))
    arrow(ax, (32, 45), (36, 45)); arrow(ax, (66, 45), (70, 45))
    arrow(ax, (84, 40), (60, 34), rad=0.1)
    arrow(ax, (50, 25), (50, 18))
    save(fig, "09_masknet_v2_bicfcpn")


# ---------------------------------------------------------------- 10) final mask
def mask_final():
    fig, ax = new_fig(14, 11.5, "ФИНАЛЬНАЯ модель маски: MaskNet-v2 bicfcpn_big "
                       "(сырой рефлектанс + замороженный DOFA-дистиллят)")
    box(ax, 2, 90, 30, 8, "vol (B,T,6,32,32)\nСЫРОЙ рефлектанс/10000\n(clip 0..1.5, без z-норм)", C_IN, 8.5)
    box(ax, 36, 90, 26, 8, "wl (6)\nB02,B03,B04,\nB08,B11,B12", C_IN, 8.5)
    box(ax, 66, 90, 32, 8, "dt (B,T) + doy (B,T,3)\nΔt в годах + sin/cos DOY\n(+ год: 0 для мозаик)", C_IN, 8.5)
    box(ax, 2, 68, 60, 18, "CnnEncoder 1.41M (ЗАМОРОЖЕН, дистилляция DOFA-B: cos 0.899):\n"
        "BandConvStem: w_c = MLP(λ_c) (1→64→64→48·3·3) → Σ_c conv3×3(x_c, w_c) → 48\n"
        "(при экспорте λ-стем свёрнут в одну dense-Conv(6→48), exact)\n"
        "ConvBlock ×4: Conv3×3 stride2 pad1 + GroupNorm(8) + GELU + 1×1 stride2 skip (residual):\n"
        "32×32(48) → 16×16(96) → 8×8(192) → 4×4(256) → 2×2(256) → GAP → 256-d на дату", C_NET, 8.5)
    box(ax, 66, 74, 32, 10, "временной вектор:\nconcat(doy, dt) = 4\n(сезон + реальные Δt)", C_FUS, 8.5)
    box(ax, 2, 55, 96, 8, "concat(256 + 4) = 260 на дату  (B,T,260)", C_FUS, 9.5, bold=True)
    box(ax, 2, 39, 30, 12, "pre-norm:\nLayerNorm(260) →\nLinear(260→128)\n"
        "Δt-норм: dt / средний\nинтервал ряда (dt[0]=0)", C_NET, 8.5)
    box(ax, 36, 39, 30, 12, "CfC_fwd и CfC_bwd (backbone 128×2):\nz = [x, h] → 2×(Linear→LeakyReLU)\n"
        "gate = σ(time_a(z)·dt + time_b(z))\nh' = (1−gate)·ff1(z) + gate·ff2(z)\n"
        "length-aware: mean по валидным T, backward с последней даты\n"
        "динамическое T (ONNX Loop + lengths), CfC на CPU", C_NET, 8)
    box(ax, 70, 39, 28, 12, "out = mean_T(concat(y_f, y_b))\n= 256-d (двунаправленность)", C_FUS, 8.5)
    box(ax, 20, 22, 60, 12, "сводки: mean_T(f) + last−first(f) по валидным датам → 512-d\n"
        "голова: LayerNorm(768) → Linear(768→256) → GELU → Dropout(0.2) → Linear(256→2), bias [1,−1]",
        C_FUS, 8.5, bold=True)
    box(ax, 40, 12, 20, 7, "logits (B,2)\n0 нет / 1 изменение", C_OUT, 9.5, bold=True)
    box(ax, 2, 0.5, 96, 9.5, "Метрики: Vozh val F1 0.929 (P 0.916 / R 0.942; p1 0.925 / p2 0.960 / p3 0.906), "
        "clean-val 0.940; GT p3 на сцене через плагин: P 0.966 / R 0.952 / F1 0.959\n"
        "Одно-контекстная модель (vozh): на big4/cal/pairs/long zero-shot без контекста слабо "
        "(R 0.166/0.001/0.207/0.116) — нужен домен-ретрейн или контекст\n"
        "Обучение: vozh 6030, энкодер заморожен, 60 эпох, lr 1.5e-4, ctx-drop 0.3; ~1.2M (CfC+ctx+голова) + 1.41M энкодер\n"
        "Скорость: сцена p3 — маска 59–65 с (энкодер DML ~0.036 мс/чип, CfC CPU); ONNX 5.5 + 4.9 МБ", C_FUS, 8)
    arrow(ax, (17, 90), (17, 86)); arrow(ax, (49, 90), (49, 86)); arrow(ax, (82, 90), (82, 86))
    arrow(ax, (62, 79), (66, 79))
    arrow(ax, (32, 68), (32, 63)); arrow(ax, (82, 74), (50, 63), rad=0.15)
    arrow(ax, (17, 55), (17, 51))
    arrow(ax, (32, 45), (36, 45)); arrow(ax, (66, 45), (70, 45))
    arrow(ax, (84, 39), (60, 34), rad=0.1)
    arrow(ax, (50, 22), (50, 19))
    save(fig, "10_mask_final_bicfcpn_big")


# ---------------------------------------------------------------- 11) final typenet
def typenet_final():
    fig, ax = new_fig(14, 11.5, "ФИНАЛЬНЫЙ TypeNet: дистиллированный DOFA-энкодер 12×768 + CfC 2×256 (3 класса)")
    box(ax, 2, 90, 30, 8, "маска → изменённые чипы\n32 px (320 м), ≤8 дат\nСЫРОЙ рефлектанс/10000", C_IN, 8.5)
    box(ax, 36, 90, 26, 8, "wl (6)\nдлины волн B02..B12", C_IN, 8.5)
    box(ax, 66, 90, 32, 8, "dt (B,T) + doy (B,T,3)\nΔt в годах + sin/cos DOY\n(+ год: 0 для мозаик)", C_IN, 8.5)
    box(ax, 2, 66, 60, 20, "typenet_encoder.onnx — CnnDofaLike 7.58M (заморожен, дистилляция DOFA-B: cos 0.866):\n"
        "BandConvStem (λ-ядра) → 48 → Conv3×3 stride2 → 768 (16×16)\n"
        "12 блоков: [dw3×3 (d=1,2,4,8 ×3) + GroupNorm(8) + GELU + pw1×1 + GroupNorm(8)] + residual + GELU\n"
        "RF ≈ 185 px (> чипа) → GroupNorm(8) → GAP → 768-d на дату", C_NET, 8.5)
    box(ax, 66, 74, 32, 10, "feats (M,T,768)\nна дату; кэш f16\nпри обучении", C_FUS, 8.5)
    box(ax, 2, 52, 96, 8, "typenet_cfc.onnx — TypeNetFeats 2.21M: x = concat(feats 768, doy 3, dt 1) = 772 на дату",
        C_FUS, 9, bold=True)
    box(ax, 6, 30, 40, 18, "CfCBiBlock ×2 (h = 256, backbone 256×2):\n"
        "u = LayerNorm(d_in) → xp = Linear(d_in→256)\n"
        "y = concat(y_f, y_b) = 512 (fwd + bwd CfC, residual,\n"
        "dt / средний интервал ряда)\n"
        "блок 2: d_in = 512", C_NET, 8.5)
    box(ax, 52, 30, 42, 18, "CfCCell: z = [x, h] → 2×(Linear(512→256)+LeakyReLU(0.1))\n"
        "gate = σ(time_a(z)·dt + time_b(z))\nh' = (1−gate)·ff1(z) + gate·ff2(z)\n"
        "mean по T → 512-d (любая длина ряда)", C_NET, 8.5)
    box(ax, 20, 17, 60, 9, "сводки: mean(feats) + (last−first)(feats) → 1536-d\n"
        "голова: LayerNorm(2048) → Linear(2048→256) → GELU → Dropout(0.2) → Linear(256→3)\n"
        "CfC: динамическое T (ONNX Loop), точные длины, CPU", C_FUS, 8.5, bold=True)
    box(ax, 38, 6, 24, 8, "logits (M,3)\nlogging / windfall / fire\n(k = argmax+1)", C_OUT, 9, bold=True)
    box(ax, 2, -3.5, 96, 8, "Метрики (val, log/wind/fire): ALL 0.870 (0.922/0.877/0.983); vozh 0.547 (0.944/0.698/-), "
        "big4 0.860 (0.816/0.830/0.934), cal 0.972 (—/—/0.972), pairs 0.920 (0.921/0.879/0.959), long 0.939 (0.927/0.889/1.000)\n"
        "GT-чипы p3: recall logging 88% / windfall 94%; no_change не классифицируется (решает маска)", C_FUS, 8)
    arrow(ax, (17, 90), (17, 86)); arrow(ax, (49, 90), (49, 86)); arrow(ax, (82, 90), (82, 86))
    arrow(ax, (62, 76), (66, 76))
    arrow(ax, (32, 66), (32, 60)); arrow(ax, (82, 74), (50, 60), rad=0.15)
    arrow(ax, (46, 47), (52, 47))
    arrow(ax, (73, 30), (60, 26), rad=0.1)
    arrow(ax, (50, 17), (50, 14))
    save(fig, "11_typenet_final")


if __name__ == "__main__":
    pipeline()
    mask_3dcnn_cfc()
    masknet_ck()
    type_fusion()
    typenet_ck()
    masknet_v2()
    typenet_v3()
    masknet_v2_cnn()
    masknet_v2_bicfcpn()
    mask_final()
    typenet_final()
    print(f"[arch] all diagrams -> {OUT}", flush=True)
