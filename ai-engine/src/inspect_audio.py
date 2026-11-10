
from pathlib import Path

import librosa
import numpy as np

# Menentukan lokasi utama proyek ai-engine
BASE_DIR = Path(__file__).resolve().parents[1]

# Lokasi folder training fan
DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MIMII_DUE"
    / "fan"
    / "fan"
    / "train"
)

# Mencari satu file audio normal dari domain source
audio_files = sorted(
    DATA_DIR.glob("section_*_source_train_normal_*.wav")
)

if not audio_files:
    raise FileNotFoundError("File audio training normal tidak ditemukan.")

audio_path = audio_files[0]

# Membaca audio tanpa mengubah sample rate aslinya
audio, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

# Menghitung durasi audio
duration = len(audio) / sample_rate

# Menampilkan informasi audio
print("=== AUDIO INFORMATION ===")
print(f"File       : {audio_path.name}")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration   : {duration:.2f} seconds")
print(f"Samples    : {len(audio)}")
print(f"Channels   : Mono")
print(f"Min Value  : {np.min(audio):.4f}")
print(f"Max Value  : {np.max(audio):.4f}")
print(f"Mean       : {np.mean(audio):.4f}")
print(f"Std        : {np.std(audio):.4f}")
