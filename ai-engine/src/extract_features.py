
from pathlib import Path

import librosa
import numpy as np


# 1. Menentukan lokasi project
BASE_DIR = Path(__file__).resolve().parents[1]

# Lokasi dataset audio normal
DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MIMII_DUE"
    / "fan"
    / "fan"
    / "train"
)

# Lokasi penyimpanan hasil ekstraksi
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "fan"
    / "preview"
)


# 2. Mencari file audio normal
audio_files = sorted(
    DATA_DIR.glob("section_*_source_train_normal_*.wav")
)

if not audio_files:
    raise FileNotFoundError(
        "File audio normal tidak ditemukan."
    )

# Mengambil satu file pertama
audio_path = audio_files[0]


# 3. Membaca audio
audio, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

if sample_rate != 16000:
    raise ValueError(
        f"Sample rate tidak sesuai: {sample_rate} Hz"
    )


# 4. Menghitung Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    fmin=0,
    fmax=8000
)


# 5. Mengubah Mel Spectrogram menjadi skala desibel
log_mel_spectrogram = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)


# 6. Memastikan tipe data float32
log_mel_spectrogram = log_mel_spectrogram.astype(
    np.float32
)


# 7. Membuat folder output
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Nama file hasil ekstraksi
output_path = (
    OUTPUT_DIR
    / f"{audio_path.stem}_logmel.npy"
)


# 8. Menyimpan matriks ke format NPY
np.save(
    output_path,
    log_mel_spectrogram
)


# 9. Menampilkan informasi hasil ekstraksi
print("=== FEATURE EXTRACTION RESULT ===")

print(f"Source File : {audio_path.name}")
print(f"Sample Rate : {sample_rate} Hz")
print(f"Feature     : Log-Mel Spectrogram")
print(f"Shape       : {log_mel_spectrogram.shape}")
print(f"Data Type   : {log_mel_spectrogram.dtype}")
print(f"Minimum     : {np.min(log_mel_spectrogram):.2f} dB")
print(f"Maximum     : {np.max(log_mel_spectrogram):.2f} dB")

print(f"\nSaved To    : {output_path}")
