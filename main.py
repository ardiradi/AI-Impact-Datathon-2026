"""
main.py — FastAPI Backend API untuk prediksi Daya Beli Score
menggunakan model RandomForestRegressor yang sudah dilatih (model.pkl).

Endpoint:
  GET  /         -> Health check / status server
  POST /predict  -> Prediksi Daya_Beli_Score berdasarkan Suku_Bunga & Indeks_Saham
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ekonomi Makro - Daya Beli Predictor",
    description="API untuk memprediksi Daya Beli Score UMKM berdasarkan indikator ekonomi makro.",
    version="1.0.0",
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schema ─────────────────────────────────────────────────────────
class MacroInput(BaseModel):
    Suku_Bunga: float
    Indeks_Saham: float


# ── Helper: Rekomendasi Strategis ────────────────────────────────────────────
def generate_rekomendasi(suku_bunga: float, indeks_saham: float, score: float) -> str:
    """Menghasilkan kalimat rekomendasi strategis untuk UMKM berdasarkan hasil prediksi."""

    if score >= 95:
        outlook = "sangat tinggi"
        saran = (
            "Ini adalah momentum terbaik untuk ekspansi bisnis. "
            "UMKM disarankan untuk menambah lini produk, memperluas jangkauan pasar digital, "
            "dan mempertimbangkan investasi pada kapasitas produksi."
        )
    elif score >= 85:
        outlook = "baik"
        saran = (
            "Kondisi daya beli cukup mendukung pertumbuhan. "
            "UMKM disarankan untuk mengoptimalkan strategi pemasaran, "
            "menjaga efisiensi operasional, dan mulai membangun cadangan modal kerja."
        )
    elif score >= 75:
        outlook = "moderat"
        saran = (
            "Daya beli masyarakat berada di level menengah. "
            "UMKM sebaiknya fokus pada retensi pelanggan, menekan biaya produksi, "
            "dan mempertimbangkan diversifikasi produk untuk menjaga stabilitas pendapatan."
        )
    else:
        outlook = "rendah"
        saran = (
            "Daya beli masyarakat sedang tertekan. "
            "UMKM disarankan untuk menerapkan strategi harga kompetitif, "
            "mengurangi pengeluaran non-esensial, dan memperkuat penjualan melalui kanal digital "
            "untuk menjangkau segmen pasar yang lebih luas."
        )

    return (
        f"Dengan Suku Bunga {suku_bunga:.2f}% dan Indeks Saham {indeks_saham:.0f}, "
        f"prediksi Daya Beli Score adalah {score:.2f} (kategori: {outlook}). "
        f"{saran}"
    )


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Endpoint sederhana untuk mengecek status server."""
    return {
        "status": "ok",
        "message": "Ekonomi Makro Prediction API is running.",
        "model": "RandomForestRegressor",
        "features": ["Suku_Bunga", "Indeks_Saham"],
        "target": "Daya_Beli_Score",
    }


@app.post("/predict")
def predict(data: MacroInput):
    """
    Menerima input Suku_Bunga dan Indeks_Saham,
    mengembalikan prediksi Daya_Beli_Score beserta rekomendasi strategis.
    """
    features = np.array([[data.Suku_Bunga, data.Indeks_Saham]])
    prediction = float(model.predict(features)[0])

    rekomendasi = generate_rekomendasi(data.Suku_Bunga, data.Indeks_Saham, prediction)

    return {
        "input": {
            "Suku_Bunga": data.Suku_Bunga,
            "Indeks_Saham": data.Indeks_Saham,
        },
        "prediksi": {
            "Daya_Beli_Score": round(prediction, 4),
        },
        "rekomendasi_strategis": rekomendasi,
    }
