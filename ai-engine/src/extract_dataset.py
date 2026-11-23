
from pathlib import Path

import librosa
import numpy as np


# 1. Lokasi project
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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "fan"
    / "train"
)


# 2. Konfigurasi ekstraksi
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64


# 3. Mencari seluruh audio normal
audio_files = sorted(
    DATA_DIR.glob("section_*_source_train_normal_*.wav")
)

if not audio_files:
    raise FileNotFoundError(
        "Dataset audio normal tidak ditemukan."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 4. Proses ekstraksi
success_count = 0
failed_count = 0

print("=== DATASET FEATURE EXTRACTION ===")
print(f"Total audio : {len(audio_files)}")
print("Processing...\n")


for index, audio_path in enumerate(audio_files, start=1):

    output_path = (
        OUTPUT_DIR
        / f"{audio_path.stem}_logmel.npy"
    )

    try:

        # Membaca audio
        audio, sr = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Sample rate tidak sesuai: {sr}"
            )

        # Menghitung Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=0,
            fmax=8000
        )

        # Mengubah menjadi Log-Mel
        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Mengubah tipe data
        log_mel = log_mel.astype(
            np.float32
        )

        # Validasi hasil
        if log_mel.shape != (64, 626):
            raise ValueError(
                f"Shape tidak sesuai: {log_mel.shape}"
            )

        # Menyimpan fitur
        np.save(
            output_path,
            log_mel
        )

        success_count += 1

    except Exception as error:

        failed_count += 1

        print(
            f"\nERROR: {audio_path.name}"
        )

        print(error)

    # Menampilkan progres
    if index % 100 == 0 or index == len(audio_files):

        print(
            f"Progress: {index}/{len(audio_files)} "
            f"| Success: {success_count} "
            f"| Failed: {failed_count}"
        )


# 5. Ringkasan hasil
print("\n=== EXTRACTION SUMMARY ===")

print(f"Total audio : {len(audio_files)}")
print(f"Success     : {success_count}")
print(f"Failed      : {failed_count}")

print(f"\nOutput directory: {OUTPUT_DIR}")
