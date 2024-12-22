# **KLASIFIKASI JENIS IKAN MENGGUNAKAN MODEL INCEPTIONV3 DAN MOBILENETV2**

## 📖 **OVERVIEW PROJECT**
Tujuan dari pembuatan proyek ini adalah untuk mengembangkan sistem yang dapat mempermudah dan mempercepat proses identifikasi jenis ikan pada hasil tambak, guna meningkatkan efisiensi dan kualitas pengelolaan tambak secara keseluruhan. Pengelolaan hasil tambak sering menghadapi tantangan dalam menyortir jenis ikan secara cepat dan akurat, terutama ketika beragam spesies seperti Black Sea Sprat, Gilt-Head Bream, Horse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, dan Trout harus diidentifikasi. Tantangan ini dapat menyebabkan penurunan efisiensi dan kualitas produk jika dilakukan secara manual. Untuk mengatasi masalah tersebut, teknologi berbasis pengolahan citra hadir sebagai solusi yang memungkinkan identifikasi dan penyortiran ikan secara otomatis. Teknologi ini tidak hanya meningkatkan efisiensi pengelolaan tambak, tetapi juga memastikan kualitas produk sesuai standar pasar, sekaligus mempermudah proses distribusi dan mendukung keberlanjutan usaha tambak.

📂 Link Dataset yang digunakan: [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data). 

---

## 🔧 **Preprocessing**
Proses preprocessing bertujuan untuk mempersiapkan dataset gambar agar siap digunakan dalam pelatihan model machine learning. Tahapan preprocessing meliputi:
1. **Resize**: Gambar diubah ukurannya menjadi 224x224 piksel.
2. **Normalization**: Nilai piksel dinormalisasi untuk mempercepat konvergensi model.
3. **Augmentation**: Teknik augmentasi seperti rotasi, flip, zoom, translasi, penyesuaian brightness, dan contrast diterapkan untuk memperkaya dataset.

Dataset dibagi menjadi tiga set utama:
- **Train**: 70% digunakan untuk melatih model.
- **Validation**: 20% digunakan untuk memantau kinerja model selama pelatihan.
- **Test**: 10% digunakan untuk mengevaluasi model setelah pelatihan.

---

## 🧠 **Model yang Digunakan**

### **1. InceptionV3**
![InceptionV3 Architecture](assets/Gambar%20Arsitektur%20InceptionV3.png)
- **Pengertian Umum**: InceptionV3 adalah salah satu model deep learning yang populer untuk tugas klasifikasi citra. Model ini dirancang dengan arsitektur mendalam yang menggunakan modul Inception untuk menangkap informasi visual dengan efisien melalui berbagai ukuran filter. InceptionV3 dioptimalkan untuk memberikan akurasi tinggi pada dataset besar sambil tetap menjaga efisiensi komputasi.
- **Hasil Pelatihan**:
  - Akurasi pelatihan: **87.47%** (epoch ke-20)
  - Akurasi validasi: **99.00%**

### **2. MobileNetV2**
![MobileNetV2 Architecture](assets/MobileNetv2.png)
- **Pengertian Umum**: MobileNetV2 adalah model deep learning yang dirancang untuk perangkat dengan daya komputasi rendah, seperti ponsel dan perangkat IoT. Model ini menggunakan blok inverted residuals dan depthwise separable convolution untuk meningkatkan efisiensi tanpa mengorbankan akurasi. MobileNetV2 sangat cocok untuk aplikasi real-time dengan latensi rendah.
- **Hasil Pelatihan**:
  - Akurasi pelatihan: **87.59%** (epoch ke-20)
  - Akurasi validasi: **99.33%**

---

## 📃 **Dependensi & Langkah Instalasi**

# Dependensi yang dibutuhkan
dependencies = ["tensorflow>=2.18.0", "joblib>=1.4.2", "scikit-learn>=1.6.0", "streamlit>=1.41.1"]

# Langkah instalasi tensorflow menggunakan pdm

# 1. Pastikan berada di dalam direktori virtual environment `.venv`
pdm info

# 2. Periksa apakah tensorflow sudah terinstal dalam virtual environment
pdm run python -m pip show tensorflow

# 3. Jika belum terinstal, jalankan perintah berikut untuk menginstal tensorflow
pdm run python -m ensurepip --upgrade
pdm run python -m pip install tensorflow

# 4. Verifikasi instalasi dengan memeriksa versi tensorflow
pdm run python -c "import tensorflow as tf; print(tf.version)"

# Struktur file proyek
- app.py: Berkas aplikasi utama yang berisi rute dan fungsi.
- klasifikasi_citra.py: Berkas penerapan dari model untuk klasifikasi dan tampilan antarmuka web.

# Menjalankan APP
- streamlit run ./src/app.py
- Akses aplikasi melalui peramban web dengan alamat: http://localhost:8501/

---

## 📊 **Evaluasi Model**
### **1. Hasil Grafik Pelatihan**
**InceptionV3**: Akurasi model meningkat secara konsisten dari 35.72% (epoch pertama) hingga 87.47% (epoch ke-20) dengan akurasi validasi mencapai 99.00%.  
**MobileNetV2**: Akurasi pelatihan meningkat dari 38.92% (epoch pertama) hingga 87.59% (epoch ke-20), dengan akurasi validasi stabil di 99.33%.  

### **2. Classification Report**
Model menunjukkan performa luar biasa dengan **precision**, **recall**, dan **f1-score** mendekati **1.00** pada sebagian besar kelas. Akurasi keseluruhan mencapai **99%**.

### **3. Confusion Matrix**
- **InceptionV3**: Hampir semua sampel berhasil diklasifikasikan dengan benar. Kesalahan kecil ditemukan, seperti pada kelas "Sea Bass" dengan 3 sampel salah klasifikasi.
- **MobileNetV2**: Performa yang hampir serupa dengan beberapa kesalahan kecil, seperti pada kelas "Striped Red Mullet".

---

## 🌐 **Local Deployment**
Aplikasi ini dilengkapi dengan antarmuka lokal untuk memprediksi jenis ikan berdasarkan input gambar.  
**Tampilan Utama**:
1. Input gambar ikan.
2. Hasil prediksi jenis ikan ditampilkan.

---

## 🚀 **Cara Menjalankan**
1. Clone repositori ini:
   ```bash
   git clone https://github.com/miftahulputra/UAP_MACHINE_LEARNING.git
