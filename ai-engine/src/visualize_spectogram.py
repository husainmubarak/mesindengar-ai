
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Menentukan lokasi dataset
BASE_DIR = Path(__file__).resolve().parents[1]

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

# Mengubah audio menjadi Short-Time Fourier Transform
stft = librosa.stft(
    audio,
    n_fft=1024,
    hop_length=256
)

# Mengubah amplitudo menjadi satuan desibel
spectrogram = librosa.amplitude_to_db(
    np.abs(stft),
    ref=np.max
)

# Menampilkan spectrogram
plt.figure(figsize=(14, 6))

librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    cmap="magma"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram - Fan Machine (Normal)")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()
