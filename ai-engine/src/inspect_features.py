
from pathlib import Path

import numpy as np


# 1. Menentukan lokasi file NPY
BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "fan"
    / "preview"
)

feature_files = sorted(
    FEATURE_DIR.glob("*_logmel.npy")
)

if not feature_files:
    raise FileNotFoundError(
        "File NPY tidak ditemukan."
    )

feature_path = feature_files[0]


# 2. Membaca matriks fitur
features = np.load(feature_path)


# 3. Menampilkan informasi matriks
print("=== FEATURE INSPECTION ===")

print(f"File       : {feature_path.name}")
print(f"Shape      : {features.shape}")
print(f"Dimensions : {features.ndim}")
print(f"Data Type  : {features.dtype}")

print("\n=== MATRIX VALUES ===")

print("First 5 rows, first 5 columns:")
print(features[:5, :5])

print("\n=== STATISTICS ===")

print(f"Mean       : {np.mean(features):.4f}")
print(f"Std        : {np.std(features):.4f}")
print(f"Minimum    : {np.min(features):.4f}")
print(f"Maximum    : {np.max(features):.4f}")
