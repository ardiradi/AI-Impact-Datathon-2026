from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Macroeconomy & UMKM Predictor API",
    description="REST API untuk memprediksi Daya Beli Masyarakat dan memberikan rekomendasi aksi UMKM.",
    version="1.0.0"
)

# Global variable untuk menyimpan model yang di-load
MODEL_PATH = "rf_dayabeli_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model {MODEL_PATH} berhasil di-load.")
        except Exception as e:
            print(f"Gagal me-load model: {e}")
    else:
        print(f"Warning: File {MODEL_PATH} tidak ditemukan. API akan menggunakan simulasi perhitungan statis.")

# Definisi skema Input JSON
class PredictionRequest(BaseModel):
    suku_bunga: float
    indeks_saham: float
    volume_transaksi_umkm: float = 12000.0  # Optional field dengan default value

# Definisi skema Output JSON
class PredictionResponse(BaseModel):
    prediksi_daya_beli_score: float
    status_daya_beli: str
    rekomendasi_aksi: str

@app.get("/")
def read_root():
    return {"message": "API Prediksi Daya Beli UMKM berjalan. Akses /docs untuk melihat dokumentasi interaktif Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict_daya_beli(data: PredictionRequest):
    # Model Random Forest kita membutuhkan 5 fitur:
    # ['Suku_Bunga', 'Indeks_Saham', 'Volume_Transaksi_UMKM', 'MA_7_Indeks_Saham', 'MA_30_Suku_Bunga']
    # Kita mengaproksimasi MA_7 dan MA_30 menggunakan input point-in-time saat ini
    features = np.array([[
        data.suku_bunga,
        data.indeks_saham,
        data.volume_transaksi_umkm,
        data.indeks_saham, # Aproksimasi MA 7 hari
        data.suku_bunga    # Aproksimasi MA 30 hari
    ]])
    
    # Lakukan prediksi menggunakan model ML yang sudah di-training, atau gunakan fallback statis jika tidak ada
    if model is not None:
        prediksi_score = float(model.predict(features)[0])
    else:
        # Fallback perhitungan kasar jika model .pkl belum di-generate
        prediksi_score = 100 - (data.suku_bunga * 5) + (data.indeks_saham - 6000) * 0.01
        
    # Pastikan rentang skor 0-100
    prediksi_score = max(0.0, min(100.0, prediksi_score))
    
    # -------------------------------------------------------------
    # ACTIONABLE INSIGHTS GENERATOR (Logic Aturan Bisnis)
    # -------------------------------------------------------------
    if prediksi_score > 70:
        status = "Tinggi"
        rekomendasi = "Daya beli sedang sangat kuat! UMKM disarankan untuk agresif melakukan campaign promosi, tawarkan strategi upselling, dan rilis stok produk premium."
    elif prediksi_score > 40:
        status = "Sedang"
        rekomendasi = "Daya beli moderat. Pertahankan operasional normal dengan fokus pada retensi pelanggan lama dan efisiensi biaya marketing."
    else:
        status = "Rendah"
        rekomendasi = "Daya beli sedang melemah (Waspada). UMKM sebaiknya fokus menjual produk kebutuhan pokok (basic needs), tawarkan paket bundling ekonomis, dan perkuat cadangan kas (likuiditas)."
        
    return PredictionResponse(
        prediksi_daya_beli_score=round(prediksi_score, 2),
        status_daya_beli=status,
        rekomendasi_aksi=rekomendasi
    )
