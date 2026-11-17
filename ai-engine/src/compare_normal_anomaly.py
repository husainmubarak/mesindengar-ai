
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np


# 1. Menentukan lokasi dataset
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MIMII_DUE"
    / "fan"
    / "fan"
    / "source_test"
)


# 2. Mencari file normal dan anomaly
normal_files = sorted(
    DATA_DIR.glob("section_*_source_test_normal_*.wav")
)

anomaly_files = sorted(
    DATA_DIR.glob("section_*_source_test_anomaly_*.wav")
)

if not normal_files or not anomaly_files:
    raise FileNotFoundError(
        "File normal atau anomaly tidak ditemukan."
    )

normal_path = normal_files[40]

# Mencari pasangan anomaly dengan section dan nomor yang sama
expected_anomaly = DATA_DIR / normal_path.name.replace(
    "_normal_", "_anomaly_"
)

anomaly_path = expected_anomaly


# 3. Fungsi untuk menghasilkan Mel Spectrogram
def extract_mel(audio_path):

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=0,
        fmax=8000
    )

    return mel, sample_rate


# 4. Ekstraksi kedua audio
normal_mel, normal_sr = extract_mel(normal_path)

anomaly_mel, anomaly_sr = extract_mel(anomaly_path)

if normal_sr != anomaly_sr:
    raise ValueError("Sample rate kedua audio berbeda.")


# 5. Menggunakan referensi desibel yang sama
common_reference = max(
    np.max(normal_mel),
    np.max(anomaly_mel)
)

normal_db = librosa.power_to_db(
    normal_mel,
    ref=common_reference
)

anomaly_db = librosa.power_to_db(
    anomaly_mel,
    ref=common_reference
)


# 6. Menampilkan perbandingan
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 6),
    sharex=True,
    sharey=True
)

image = librosa.display.specshow(
    normal_db,
    sr=normal_sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel",
    fmin=0,
    fmax=8000,
    cmap="magma",
    vmin=-80,
    vmax=0,
    ax=axes[0]
)

librosa.display.specshow(
    anomaly_db,
    sr=anomaly_sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel",
    fmin=0,
    fmax=8000,
    cmap="magma",
    vmin=-80,
    vmax=0,
    ax=axes[1]
)

axes[0].set_title("Normal")
axes[1].set_title("Anomaly")

axes[0].set_ylabel("Frequency (Hz, Mel scale)")
axes[1].set_ylabel("Frequency (Hz, Mel scale)")

fig.colorbar(
    image,
    ax=axes,
    format="%+2.0f dB"
)

plt.suptitle("Normal vs Anomaly - Fan Machine")
plt.tight_layout()
plt.show()


# 7. Menampilkan nama file yang dibandingkan
print("=== FILE COMPARISON ===")
print(f"Normal  : {normal_path.name}")
print(f"Anomaly : {anomaly_path.name}")
