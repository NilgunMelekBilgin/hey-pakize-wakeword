# 🎙️ Hey Pakize — Wake Word Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/librosa-0.10+-green" />
  <img src="https://img.shields.io/badge/CustomTkinter-UI-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **"Hey Pakize"** uyandırma kelimesi (wake word) algılama sistemi. Gerçek zamanlı mikrofon girişi üzerinde sliding-window yöntemi ile çalışan, CNN tabanlı derin öğrenme modeli ve masaüstü GUI arayüzü içerir.

---

## 📄 Lisans

Bu proje şu anda resmi bir lisans ile yayınlanmamıştır.
Projeyi kopyalamadan veya dağıtmadan önce proje sahipleri ile iletişime geçmeniz gerekmektedir.

### 👥 Proje Sahipleri
- Nilgün Melek Bilgin  
- Gamze Özdemir  
- Çiğdem Kurt  
- Sıla Dertli  
- Hanife Çilingir


##  İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Klasör Yapısı](#-klasör-yapısı)
- [Teknik Detaylar](#-teknik-detaylar)

---

##  Proje Hakkında

Bu proje, özel bir uyandırma kelimesi olan **"Hey Pakize"** sesini mikrofon akışından gerçek zamanlı olarak tanıyan bir sistem geliştirmek amacıyla oluşturulmuştur. Sistem; bir CNN modeli, MFCC+Delta özellik çıkarımı ve sliding-window gerçek zamanlı ses işleme pipeline'ından oluşmaktadır.

Model, Google Colab üzerinde GPU (T4) kullanılarak eğitilmiş; masaüstü arayüzü ise `customtkinter` kütüphanesi ile Python'da geliştirilmiştir.

---

## ✨ Özellikler

- 🎤 **Gerçek zamanlı mikrofon dinleme** — sounddevice ile sürekli ses akışı
- 🪟 **Sliding-window algılama** — 1.5 sn pencere, 0.3 sn adım boyutu
- 🧠 **CNN tabanlı model** — MFCC + Delta + Delta² özellik vektörü
- 🎚️ **Ayarlanabilir eşik** — Arayüzden anlık threshold kontrolü
- ⏱️ **Cooldown mekanizması** — Çift tetiklenmeyi önler (2 sn bekleme)
- 🖥️ **Modern masaüstü GUI** — Dark tema, log paneli, test butonu
- 🔄 **Dengeli veri augmentasyonu** — Noise + Time-shift (pozitif & negatif)

---

## 🏗️ Sistem Mimarisi

```
Mikrofon (16kHz)
      │
      ▼
 Ses Tamponu (Ring Buffer — 1.5 sn)
      │  ← Her 0.3 sn'de bir sliding step
      ▼
 process_audio()
   - Trim (top_db=20)
   - Pad / Crop → 24000 sample
   - Normalize
      │
      ▼
 extract_mfcc()
   - MFCC (40 katsayı)
   - Delta
   - Delta² 
   → Shape: (120, 47, 1)
      │
      ▼
 CNN Model (hey_pakize_model_v3.h5)
      │
      ▼
 Sigmoid Çıkışı → Threshold (≥ 0.63)
      │
      ▼
 ✅ "Hey Pakize Algılandı!" / ❌ Negatif
```

---

## 📁 Klasör Yapısı

```
├── modeleğitimkodu_ve_modeller/
│   ├── hey_pakize_model_v3.h5
│   ├── hey_pakize_model_v3.h5_kod.ipynb
│   ├── hey_pakize_model_v3_meta.json
│   └── model_requirements.txt
│
└── proje_arayuz_kodları/
    ├── arayüz_requirements.txt
    └── hey_pakize/
        ├── sliding_demo.py
        │
        └── model/
            ├── hey_pakize_model_v3.h5
            └── hey_pakize_model_v3_meta.json
---

## 🔬 Teknik Detaylar

### Ses İşleme Pipeline'ı

```python
# 1. Yükle ve trim et
y, _ = librosa.effects.trim(y, top_db=20)

# 2. Uzunluk normalize (24000 sample = 1.5 sn @ 16kHz)
y = np.pad(y, ...) veya y[:24000]

# 3. Genlik normalize
y = y / np.max(np.abs(y))

# 4. Özellik çıkar
mfcc   = librosa.feature.mfcc(y, sr=16000, n_mfcc=40)
delta  = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
features = np.vstack([mfcc, delta, delta2])  # shape: (120, 47)
```

### Sliding Window Mekanizması

```
Ring Buffer (24000 sample = 1.5 sn)
│
├── Audio Callback: her 8000 sample'da bir tetiklenir (0.3 sn)
│   → buffer'ı sola kaydır, yeni chunk'ı sona ekle
│
└── Predict Thread: her 0.3 sn'de bir buffer snapshot al → inference
```

---

## 📊 Model Performansı

| Metrik | Değer |
|--------|-------|
| Eğitim Doğruluğu | ~%94+ |
| Validasyon Doğruluğu | ~%90+ |
| Threshold | 0.63 |
| Gecikme (inference) | < 100ms |
| Cooldown | 2 saniye |

---

