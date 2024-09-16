# Menggunakan image Python sebagai base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt .

# Menginstal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode ke dalam container
COPY . .

# Perintah default untuk menjalankan aplikasi
CMD ["python", "main.py"]
