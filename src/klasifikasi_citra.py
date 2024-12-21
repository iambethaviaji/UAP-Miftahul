import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st

# Judul aplikasi
st.title("Klasifikasi Citra Jenis Ikan")

# Deskripsi aplikasi
st.markdown(
    """
    <div style="text-align: justify; font-size: 18px; line-height: 1.8; color: #333333; padding: 20px; background: #4A90E2; 
    border-radius: 12px; box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);">
        Pengelolaan hasil tambak sering menghadapi tantangan dalam menyortir jenis ikan secara cepat dan akurat, 
        terutama ketika beragam spesies seperti <b style="color: #fff;">Black Sea Sprat</b>, <b style="color: #fff;">Gilt-Head Bream</b>, 
        <b style="color: #fff;">Horse Mackerel</b>, <b style="color: #fff;">Red Mullet</b>, <b style="color: #fff;">Red Sea Bream</b>, 
        <b style="color: #fff;">Sea Bass</b>, <b style="color: #fff;">Shrimp</b>, <b style="color: #fff;">Striped Red Mullet</b>, 
        dan <b style="color: #fff;">Trout</b> harus diidentifikasi. Tantangan ini dapat menyebabkan penurunan efisiensi dan kualitas produk jika dilakukan 
        secara manual. Untuk mengatasi masalah tersebut, teknologi berbasis pengolahan citra hadir sebagai solusi 
        yang memungkinkan identifikasi dan penyortiran ikan secara otomatis. Teknologi ini tidak hanya meningkatkan 
        efisiensi pengelolaan tambak, tetapi juga memastikan kualitas produk sesuai standar pasar, sekaligus 
        mempermudah proses distribusi dan mendukung keberlanjutan usaha tambak.
    </div>
    """,
    unsafe_allow_html=True,
)

# Tambahkan CSS untuk latar belakang dan elemen lainnya agar lebih menarik
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;  /* Warna latar belakang keseluruhan */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .stMarkdown b {
            font-weight: bold;
            color: #FFD700;  /* Warna emas untuk teks yang ditebalkan */
        }
        .stTitle {
            text-align: center;
            color: #4A90E2;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 40px;
            text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
        }
        .stButton button {
            background-color: #4A90E2;  /* Mengganti warna tombol menjadi biru */
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton button:hover {
            background-color: #357ABD;  /* Biru lebih gelap saat hover */
            transform: translateY(-2px);
        }
        .stButton button:active {
            background-color: #1E73B3;  /* Biru lebih terang saat aktif */
            transform: translateY(2px);
        }
        .stFileUploader {
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .stFileUploader input {
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Fungsi prediksi
def predict(uploaded_image, model_path):
    # Daftar kelas
    class_names = [
        "Black Sea Sprat",
        "Gilt-Head Bream",
        "Horse Mackerel",
        "Red Mullet",
        "Red Sea Bream",
        "Sea Bass",
        "Shrimp",
        "Striped Red Mullet",
        "Trout",
    ]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Muat model
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])
    return class_names[np.argmax(score)], 100 * np.max(score)


# Pilihan model
model_option = st.selectbox(
    "Pilih model untuk prediksi:", ("InceptionV3", "MobileNetV2")
)

# Tentukan path model berdasarkan pilihan
if model_option == "InceptionV3":
    model_path = Path(__file__).parent / "Model/Image/InceptionV3/model.h5"
else:
    model_path = Path(__file__).parent / "Model/Image/MobileNetV2/model.h5"

# Komponen file uploader untuk banyak file
uploads = st.file_uploader(
    "Unggah citra untuk mendapatkan hasil prediksi",
    type=["png", "jpg"],
    accept_multiple_files=True,
    label_visibility="collapsed",  # Mengurangi tampilan label
)

# Tombol prediksi
if st.button("Prediksi", type="primary"):
    if uploads:
        st.subheader("Hasil Prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(
                upload,
                caption=f"Citra yang diunggah: {upload.name}",
                use_container_width=True,
            )

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                try:
                    label, confidence = predict(upload, model_path)
                    st.write(f"**Label:** {label}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
