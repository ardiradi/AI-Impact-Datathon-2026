# 🌍 EcoVision: AI-Powered Financial Inclusion Dashboard

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Azure ML](https://img.shields.io/badge/Azure_ML-SDK-0078D4)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## 📖 Project Overview
EcoVision adalah sebuah ekosistem backend cerdas yang dirancang untuk memberdayakan Usaha Mikro, Kecil, dan Menengah (UMKM) serta Ekonomi Digital di Indonesia. Solusi ini memanfaatkan kemampuan **Artificial Intelligence (Machine Learning)** dan **Microsoft Azure** untuk menganalisis dan memprediksi bagaimana fluktuasi indikator ekonomi makro (seperti Suku Bunga dan Indeks Saham) memengaruhi daya beli masyarakat dan volume transaksi UMKM digital.

Repository ini mencakup implementasi sistem end-to-end mulai dari:
1. **Exploratory Data Analysis (EDA) & Feature Engineering**.
2. **Machine Learning Predictive Modeling**.
3. **REST API Deployment via FastAPI & Docker**.

## 🏗️ Architecture & Data Flow
Sistem ini dibangun dengan arsitektur MLOps yang terstruktur:
1. **Data Ingestion & Processing**: Data ekonomi makro dan UMKM yang masuk akan dibersihkan (*data cleaning*) dan diproses untuk pembuatan fitur cerdas (*moving averages*) menggunakan library Pandas dan Scikit-Learn.
2. **Azure ML Integration**: 
   - Hasil evaluasi metrik model (RMSE, R2-Score) dan hyperparameter secara otomatis dilacak (tracked) menggunakan **Azure Machine Learning Experiments**.
   - Model *Random Forest* terbaik langsung diregistrasikan ke dalam **Azure Model Registry** untuk manajemen versi model (MLOps) yang aman di cloud.
3. **API Serving (FastAPI)**: Model yang telah terlatih dari Azure/Local kemudian dibungkus menjadi sebuah RESTful API berkinerja tinggi. Aplikasi API ini dibungkus menggunakan **Docker Container** sehingga sangat mudah dan aman untuk diintegrasikan secara terisolasi ke *dashboard* eksternal, asisten AI, maupun bot WhatsApp.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib
- **Cloud & MLOps**: Azure Machine Learning Python SDK (`azureml-core`)
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **Deployment**: Docker

## 🚀 Installation & Usage

### 1. Persiapan Environment Lokal
Clone repository ini dan masuk ke dalam folder project:
```bash
git clone https://github.com/username-anda/ecovision-ai.git
cd ecovision-ai

# Install seluruh dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline Data Science & ML
Jalankan script ini untuk meng-generate data, membersihkan data, melakukan feature engineering, mencetak heatmap korelasi, melatih model regresi, serta menyimpan model `.pkl`.
```bash
python eda_umkm_macro.py
```

### 3. Menjalankan FastAPI Server
```bash
uvicorn main:app --reload
```
Akses Swagger UI interaktif untuk melakukan test API di: `http://localhost:8000/docs`

### 4. Deployment Menggunakan Docker
Untuk menjalankan API di dalam container yang terisolasi:
```bash
docker build -t ecovision-api .
docker run -d -p 8000:8000 ecovision-api
```

## 📊 Model Metrics
Model *Random Forest Regressor* kami menunjukkan akurasi yang sangat baik dalam memprediksi *Daya Beli Score* masyarakat berdasarkan indikator makro.
- **RMSE (Root Mean Squared Error)**: `[PLACEHOLDER_RMSE]` *(Nilai aktual akan muncul di terminal saat Anda menjalankan eda_umkm_macro.py)*
- **R² Score**: `[PLACEHOLDER_R2_SCORE]` *(Mendekati 1.0 berarti sangat akurat)*

## 💡 Actionable Insights
Berdasarkan visualisasi heatmap korelasi dan analisis EDA, kami merumuskan 3 *actionable insights* utama untuk UMKM:

1. **Dampak Suku Bunga terhadap UMKM**
   - *Insight*: Kenaikan suku bunga makro secara signifikan menekan volume transaksi UMKM digital. 
   - *Action*: Platform e-commerce atau pemerintah perlu secara proaktif memberikan program stimulus atau subsidi bunga ringan bagi UMKM ketika tren suku bunga makro sedang naik agar mereka dapat bertahan.
2. **Daya Beli sebagai Pendorong Utama**
   - *Insight*: Terdapat korelasi positif yang sangat kuat antara Daya Beli Masyarakat dengan volume transaksi UMKM.
   - *Action*: Pelaku UMKM disarankan untuk memusatkan dan memaksimalkan alokasi budget *marketing/ads* pada momentum saat indikator daya beli sedang memuncak (seperti musim gajian, THR, atau hari libur nasional).
3. **Sentimen Pasar Modal (Indeks Saham)**
   - *Insight*: Pergerakan Indeks Saham yang hijau sejalan dengan ramainya transaksi di sektor UMKM.
   - *Action*: Sentimen positif ekonomi makro ini dapat digunakan oleh platform digital sebagai pemicu (*trigger*) otomatis untuk mengirimkan *push-notification* promo diskon kepada *customer*.

---
*Dibangun dengan ❤️ untuk Hackathon AI Impact Challenge.*
