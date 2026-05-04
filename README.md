# 🎭 Deepfake Face Detector

> **ML Intensive · Yandex Academy · Spring 2026**  

[![F1 Score](https://img.shields.io/badge/F1--score-0.9942-10B981?style=flat-square)](.)
[![Top-3](https://img.shields.io/badge/Leaderboard-TOP--3-F59E0B?style=flat-square)](.)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=flat-square)](.)

Бинарная классификация изображений лиц: **реальное (Flickr) vs синтетическое (StyleGAN)**.  
Задача финального проекта курса по компьютерному зрению Yandex Academy.

---

## 📊 Результаты

| Модель | Подход | F1-score |
|--------|--------|----------|
| RegNet | Классическая CNN | ~0.958 |
| RGB + FFT | Частотный анализ | ~0.965 |
| ConvNeXt | Современная CNN | ~0.970 |
| RGB + FFT + Laplacian | Тройной поток | ~0.972 |
| EfficientNetB3 | MBConv + SE | 0.9882 |
| DualStreamNetV2 | Cross-Stream Fusion + SWA | ~0.983 |
| HFDualCNN | K-Fold + SBI | ~0.984 |
| RobustSRM MIL | SRM-фильтры + Top-K MIL | ~0.977 |
| IR-CFA-CCNet | CFA-форензика | ~0.979 |
| RGBFreqNet | fft phase | ~0.963 |
| **EfficientNetV5** ⭐ | OOF + 384px + Full Fit | **0.9942** |
| Stacking Ensemble | 10 моделей · 2 фазы | **TOP-1** |

---

## 🧠 Ключевые идеи

**1. Частотный анализ.** StyleGAN оставляет паразитные паттерны в FFT-спектре.  
Мы добавили FFT и лапласиан как параллельные потоки к RGB.

**2. Форензика.** SRM-фильтры (из стеганографии) и CFA-анализ (Bayer-паттерн камеры)  
улавливают то, что пропускают обычные CNN.

**3. OOF + финальный файнтюн.** 5-fold OOF даёт честную оценку и правильный порог.  
Дообучение на 100% данных выжимает максимум.

**4. Двухфазный стекинг.** 1023 комбинации подмножеств → топ-50 → LightGBM/RF/GBM.  
Мета-модель учится кому доверять на каждом конкретном примере.

---

## 📁 Структура репозитория

notebooks/

├── 01_regnet.ipynb               — базовая CNN, отправная точка

├── 02_rgb_fft.ipynb              — RGB + FFT двухпоточный детектор

├── 03_convnext.ipynb             — современная CNN архитектура

├── 04_rgb_fft_lap.ipynb          — тройной поток: RGB + FFT + Laplacian

├── 05_efficientnet_b3.ipynb      — EfficientNetB3 ручная реализация

├── 06_dual_stream_v2.ipynb       — Cross-Stream Fusion + SWA

├── 07_hf_dual_cnn_kfold_sbi.ipynb — K-Fold + SBI аугментация

├── 08_robust_srm_mil.ipynb       — SRM-фильтры + Top-K MIL пулинг

├── 09_ir_cfa_ccnet.ipynb         — CFA форензик анализ

├── 10_rgbfreq_net.ipynb         — Улучшенная частотная модель

├── 11_efficientnet_v5.ipynb      — лучшая модель, F1=0.9942

├── 12_ensemble_top2_predict.ipynb — предсказания для ансамбля

├── 13_stacking_v2.ipynb          — двухфазный стекинг

└── 14_visualizations.ipynb       — все визуализации для презентации

docs/

├── project_report.md             — подробный отчёт о работе

└── presentation.pptx             — презентация для защиты

---

## 🚀 Как запустить

Все блокноты рассчитаны на **Google Colab** с GPU (A100 или T4).

1. Открой любой блокнот в Colab
2. Смонтируй Google Drive с датасетом:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Укажи путь к архиву датасета (переменная `ZIP_PATH` в начале блокнота)
4. Запусти все ячейки

**Датасет:** [Google Drive →](https://drive.google.com/file/d/1xNCJLd7me2Fqno2QhSJ1zjFteWYcUpcV/view?usp=sharing)
**Веса моделей:** [Google Drive →](https://drive.google.com/drive/folders/1QFX5MSzo861ivmZBmwiCte3N4FnoZUKk?usp=sharing)
