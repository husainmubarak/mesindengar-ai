
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Menentukan lokasi proyek
BASE_DIR = Path(__file__).resolve().parents[1]

# Menentukan folder dataset fan
DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MIMII_DUE"
    / "fan"
    / "fan"
    / "train"
)

# Mengambil satu file audio normal
audio_files = sorted(
    DATA_DIR.glob("section_*_source_train_normal_*.wav")
)

if not audio_files:
    raise FileNotFoundError("File audio tidak ditemukan.")

audio_path = audio_files[0]

# Membaca audio
audio, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

# Membuat sumbu waktu
time = np.arange(len(audio)) / sample_rate

# Membuat visualisasi waveform
plt.figure(figsize=(14, 5))

plt.plot(time, audio)

plt.title("Waveform - Fan Machine (Normal)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.xlim(0, len(audio) / sample_rate)
plt.grid(True)

plt.tight_layout()
plt.show()
