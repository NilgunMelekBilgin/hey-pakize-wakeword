import os
import json
import threading
import time
from pathlib import Path
from collections import deque

import customtkinter as ctk
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# TEMA
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# DOSYA YOLLARI
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "hey_pakize_model_v3.h5"
META_PATH  = BASE_DIR / "model" / "hey_pakize_model_v3_meta.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Meta dosyası bulunamadı: {META_PATH}")


# META YÜKLE
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

TARGET_SR       = int(meta["target_sr"])           # 16000
TARGET_DURATION = float(meta["target_duration"])   # 1.5
N_MFCC          = int(meta["n_mfcc"])              # 40
THRESHOLD = float(meta.get("threshold", 0.63))
TARGET_LENGTH   = int(TARGET_SR * TARGET_DURATION) # 24000

# Sliding window adım boyutu: 0.3 sn → her 0.3 sn'de bir tahmin
STEP_DURATION = 0.30
STEP_LENGTH   = int(TARGET_SR * STEP_DURATION)     # 8000


# MODEL YÜKLE
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# SES HAZIRLAMA  )
def process_audio_array(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32).flatten()
    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) == 0:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))
    else:
        y = y[:TARGET_LENGTH]

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y.astype(np.float32)

# MFCC + DELTA + DELTA2  
def extract_mfcc_from_signal(y: np.ndarray) -> np.ndarray:
    mfcc   = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


# TAHMİN FONKSİYONU
def predict_from_audio_array(audio_array: np.ndarray):
    sig  = process_audio_array(audio_array)
    feat = extract_mfcc_from_signal(sig)
    x    = feat[np.newaxis, ..., np.newaxis]          # (1, 120, T, 1)
    prob = float(model.predict(x, verbose=0)[0][0])
    result = 1 if prob >= THRESHOLD else 0
    return result, f"%{prob * 100:.1f}", prob


# ARAYÜZ
# =========================================================
class HeyPakizeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hey Pakize — Uyandırma Sistemi")
        self.geometry("900x600")
        self.resizable(False, False)

        self.listening      = False
        self._stream        = None
        self._buffer        = np.zeros(TARGET_LENGTH, dtype=np.float32)
        self._buffer_lock   = threading.Lock()
        self._predict_thread = None

        # Cooldown: algılama sonrası 2 sn boyunca tekrar tetiklenmesin
        self._last_detect_time = 0.0
        self.COOLDOWN_SEC      = 2.0

        self._build_ui()

   
    # UI OLUŞTURMA
    def _build_ui(self):
        # Başlık
        ctk.CTkLabel(
            self, text="HEY PAKİZE UYANDIRMA SİSTEMİ",
            font=("Arial", 24, "bold")
        ).pack(pady=20)

        # Eşik etiketi
        self.thr_label = ctk.CTkLabel(
            self, text=f"Eşik: {THRESHOLD:.2f}", font=("Arial", 13))
        self.thr_label.pack(pady=(4, 0))

        # Slider
        self.thr_slider = ctk.CTkSlider(
            self, from_=0.3, to=0.99,
            number_of_steps=69,
            command=self._on_threshold_change,
            width=400
        )
        self.thr_slider.set(THRESHOLD)
        self.thr_slider.pack(pady=(2, 8))

       # Durum satırları
        self.status_label = ctk.CTkLabel(
        self, text="Durum: Bekleniyor", font=("Arial", 18))
        self.status_label.pack(pady=6)

        self.mic_label = ctk.CTkLabel(
            self, text="Mikrofon: Pasif", font=("Arial", 16))
        self.mic_label.pack(pady=4)

        self.result_label = ctk.CTkLabel(
            self, text="Sonuç: -", font=("Arial", 16))
        self.result_label.pack(pady=4)

        self.confidence_label = ctk.CTkLabel(
            self, text="Güven: -", font=("Arial", 16))
        self.confidence_label.pack(pady=4)

        ctk.CTkLabel(
            self, text=f"Eşik: {THRESHOLD:.3f}  |  Pencere: {TARGET_DURATION}s  |  Adım: {STEP_DURATION}s",
            font=("Arial", 13), text_color="gray"
        ).pack(pady=4)

        # Butonlar
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=16)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="Dinlemeyi Başlat",
            command=self.start_listening, width=170, height=40)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="Dinlemeyi Durdur",
            command=self.stop_listening, width=170, height=40)
        self.stop_btn.grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            btn_frame, text="Test Algılama",
            command=self._fake_detect, width=170, height=40
        ).grid(row=0, column=2, padx=10)

        # Log
        ctk.CTkLabel(
            self, text="Sistem Logları", font=("Arial", 16, "bold")
        ).pack(pady=(10, 4))

        self.log_box = ctk.CTkTextbox(
            self, width=800, height=200, font=("Consolas", 12))
        self.log_box.pack(pady=8)
        self.log_box.configure(state="disabled")

        self._log("Sistem hazır.")
        self._log(f"Model yüklendi  |  Threshold: {THRESHOLD:.3f}")
        self._log("Sliding-window modu aktif (pencere=1.5s, adım=0.5s)")

    # ----------------------------------------------------------
    # LOG YARDIMCILARI
    # ----------------------------------------------------------
    def _log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _safe_log(self, msg: str):
        self.after(0, lambda: self._log(msg))

    def _on_threshold_change(self, value):
        global THRESHOLD
        THRESHOLD = round(float(value), 2)
        self.thr_label.configure(text=f"Eşik: {THRESHOLD:.2f}")
        self._safe_log(f"Eşik güncellendi: {THRESHOLD:.2f}") 
    # ----------------------------------------------------------
    # DİNLEME BAŞLAT / DURDUR
    # ----------------------------------------------------------
    def start_listening(self):
        if self.listening:
            self._log("Sistem zaten dinliyor.")
            return

        self.listening = True
        with self._buffer_lock:
            self._buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)

        self.status_label.configure(text="Durum: Dinleniyor")
        self.mic_label.configure(text="Mikrofon: Aktif")
        self._log("Dinleme başlatıldı (sliding window).")

        # Ses akışını aç
        self._stream = sd.InputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
            blocksize=STEP_LENGTH,      # her callback'te 0.5 sn veri gelir
            callback=self._audio_callback
        )
        self._stream.start()

        # Tahmin döngüsünü ayrı thread'de çalıştır
        self._predict_thread = threading.Thread(
            target=self._predict_loop, daemon=True)
        self._predict_thread.start()

    def stop_listening(self):
        self.listening = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self.status_label.configure(text="Durum: Bekleniyor")
        self.mic_label.configure(text="Mikrofon: Pasif")
        self.result_label.configure(text="Sonuç: -")
        self.confidence_label.configure(text="Güven: -")
        self.geometry("900x600")
        self._log("Dinleme durduruldu.")

    # ----------------------------------------------------------
    # SLIDING WINDOW CALLBACK  (ses thread'inden çağrılır)
    # Her 0.5 sn'de bir STEP_LENGTH sample gelir;
    # buffer'ı sola kaydır ve yeni chunk'ı sona ekle.
    # ----------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0]
        with self._buffer_lock:
            self._buffer = np.roll(self._buffer, -len(chunk))
            self._buffer[-len(chunk):] = chunk

    # ----------------------------------------------------------
    # TAHMİN DÖNGÜSÜ  (ayrı thread)
    # Her 0.5 sn'de bir buffer'ın anlık kopyasını işle.
    # ----------------------------------------------------------
    def _predict_loop(self):
        while self.listening:
            time.sleep(STEP_DURATION)

            with self._buffer_lock:
                snapshot = self._buffer.copy()

            try:
                result, conf_text, prob = predict_from_audio_array(snapshot)
                self._safe_log(f"Skor: {prob:.4f}")

                now = time.time()
                if result == 1 and (now - self._last_detect_time) > self.COOLDOWN_SEC:
                    self._last_detect_time = now
                    self._safe_log(f"✅ Wake word algılandı! Güven: {conf_text}")
                    self.after(0, lambda c=conf_text: self._on_detected(c))
                else:
                    self.after(0, lambda c=conf_text: self._on_not_detected(c))

            except Exception as e:
                self._safe_log(f"Tahmin hatası: {e}")

    # ----------------------------------------------------------
    # UI GÜNCELLEME
    # ----------------------------------------------------------
    def _on_detected(self, conf_text: str):
        self.geometry("960x660")
        self.status_label.configure(text="✅ Hoş Geldiniz!")
        self.result_label.configure(text="Sonuç: Hey Pakize Algılandı")
        self.confidence_label.configure(text=f"Güven: {conf_text}")

    def _on_not_detected(self, conf_text: str):
        self.status_label.configure(text="Durum: Dinleniyor")
        self.result_label.configure(text="Sonuç: Negatif")
        self.confidence_label.configure(text=f"Güven: {conf_text}")

    def _fake_detect(self):
        self._on_detected("%100.0")
        self._safe_log("Test algılama çalıştırıldı.")


# =========================================================
# GİRİŞ NOKTASI
# =========================================================
if __name__ == "__main__":
    app = HeyPakizeApp()
    app.mainloop()