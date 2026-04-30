# Menggunakan base image Python versi 3.10 yang ringan
FROM python:3.10-slim

# Menentukan working directory di dalam container
WORKDIR /app

# Menyalin requirements.txt terlebih dahulu (untuk optimasi caching layer Docker)
COPY requirements.txt .

# Menginstal semua dependencies tanpa menyimpan cache untuk mengecilkan ukuran image
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh sisa source code dan file .pkl model ke dalam container
COPY . .

# Mengekspos port 8000 yang akan di-listen oleh uvicorn
EXPOSE 8000

# Perintah utama untuk menjalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
