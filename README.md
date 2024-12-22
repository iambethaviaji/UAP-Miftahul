# **KLASIFIKASI JENIS IKAN MENGGUNAKAN MODEL INCEPTIONV3 DAN MOBILENETV2**

## 📖 **Deskripsi Proyek**
Proyek ini bertujuan untuk mengembangkan sistem yang dapat mempermudah dan mempercepat proses identifikasi jenis ikan pada hasil tambak. Sistem ini diharapkan mampu meningkatkan efisiensi dan kualitas pengelolaan tambak dengan teknologi berbasis pengolahan citra. Tantangan dalam penyortiran ikan secara manual, terutama ketika terdapat beragam spesies, sering kali menyebabkan penurunan efisiensi dan kualitas produk. Dengan menggunakan teknologi ini, identifikasi ikan seperti:
- Black Sea Sprat
- Gilt-Head Bream
- Horse Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Shrimp
- Striped Red Mullet
- Trout  
dapat dilakukan secara otomatis.

Teknologi ini tidak hanya meningkatkan efisiensi pengelolaan tambak tetapi juga memastikan kualitas produk sesuai standar pasar, mempermudah distribusi, dan mendukung keberlanjutan usaha tambak.

---

## 📂 **Dataset**
Dataset yang digunakan adalah **A Large Scale Fish Dataset**, yang terdiri atas **9.000 gambar** yang terbagi menjadi:
- **70%** Training Set
- **20%** Validation Set
- **10%** Testing Set  

Setiap set terdiri dari 9 label kelas:  
Black Sea Sprat, Gilt-Head Bream, Horse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, dan Trout.  

**Link Dataset**: [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data)

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
- Model ini di-fine-tuning dengan dataset ikan untuk tugas klasifikasi.
- **Hasil pelatihan**:
  - Akurasi pelatihan: **87.47%** (epoch ke-20)
  - Akurasi validasi: **99.00%**

### **2. MobileNetV2**
- Model yang lebih ringan untuk kebutuhan real-time.
- **Hasil pelatihan**:
  - Akurasi pelatihan: **87.59%** (epoch ke-20)
  - Akurasi validasi: **99.33%**

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
