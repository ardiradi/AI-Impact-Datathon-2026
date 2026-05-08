"""
train.py — Script untuk melatih model Machine Learning (RandomForestRegressor)
menggunakan dataset sintesis Ekonomi Makro.

Fitur   : Suku_Bunga, Indeks_Saham
Target  : Daya_Beli_Score

Output  : model.pkl (model yang sudah dilatih)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── 1. Membuat Dataset Sintesis ──────────────────────────────────────────────
np.random.seed(42)
n_samples = 500

suku_bunga = np.random.uniform(3.0, 12.0, n_samples)        # persen (%)
indeks_saham = np.random.uniform(4000, 7500, n_samples)      # poin indeks

# Daya_Beli_Score dimodelkan sebagai fungsi dari kedua fitur + noise
# Asumsi: suku bunga tinggi → daya beli turun, indeks saham tinggi → daya beli naik
daya_beli_score = (
    80
    - 2.5 * suku_bunga
    + 0.005 * indeks_saham
    + np.random.normal(0, 3, n_samples)
)

df = pd.DataFrame({
    "Suku_Bunga": suku_bunga,
    "Indeks_Saham": indeks_saham,
    "Daya_Beli_Score": daya_beli_score,
})

print("=" * 55)
print("[DATA]  Dataset Ekonomi Makro (Sintesis)")
print("=" * 55)
print(df.describe().round(2))
print()

# ── 2. Split Data ────────────────────────────────────────────────────────────
X = df[["Suku_Bunga", "Indeks_Saham"]]
y = df["Daya_Beli_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f">> Training samples : {len(X_train)}")
print(f">> Testing  samples : {len(X_test)}")
print()

# ── 3. Melatih Model ─────────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 4. Evaluasi ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 55)
print("[EVAL]  Hasil Evaluasi Model (Test Set)")
print("=" * 55)
print(f"   MAE  : {mae:.4f}")
print(f"   MSE  : {mse:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   R²   : {r2:.4f}")
print()

# ── 5. Simpan Model ──────────────────────────────────────────────────────────
model_path = "model.pkl"
joblib.dump(model, model_path)
print(f"[OK]  Model berhasil disimpan ke '{model_path}'")
