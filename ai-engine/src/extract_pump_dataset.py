
from pathlib import Path

import librosa
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MIMII_DUE"
    / "pump"
    / "pump"
    / "train"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "pump"
    / "train"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
FMIN = 0
FMAX = 8000

audio_files = sorted(
    INPUT_DIR.glob("*_source_train_normal_*.wav")
)

success_count = 0
failed_count = 0

print("=== PUMP FEATURE EXTRACTION ===")
print(f"Total audio: {len(audio_files)}")

for index, audio_path in enumerate(audio_files, start=1):
    try:
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        log_mel = log_mel.astype(np.float32)

        if log_mel.shape != (64, 626):
            raise ValueError(
                f"Invalid shape: {log_mel.shape}"
            )

        output_path = (
            OUTPUT_DIR
            / f"{audio_path.stem}_logmel.npy"
        )

        np.save(output_path, log_mel)

        success_count += 1

        if index % 500 == 0:
            print(f"Processed: {index}/{len(audio_files)}")

    except Exception as error:
        failed_count += 1
        print(f"Failed: {audio_path.name}")
        print(f"Reason: {error}")

print("\n=== EXTRACTION SUMMARY ===")
print(f"Total audio : {len(audio_files)}")
print(f"Success     : {success_count}")
print(f"Failed      : {failed_count}")
