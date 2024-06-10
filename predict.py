import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from io import BytesIO
import os
from tqdm import tqdm

# Fungsi untuk ekstraksi fitur audio
def Feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_scaled = np.mean(zcr.T, axis=0).reshape(1, -1)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_scaled = np.mean(spectral_centroid.T, axis=0).reshape(1, -1)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spectral_bandwidth_scaled = np.mean(spectral_bandwidth.T, axis=0).reshape(1, -1)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_stft_scaled = np.mean(chroma_stft.T, axis=0).reshape(1, -1)

    # Root Mean Square Energy (RMSE)
    rmse = librosa.feature.rms(y=audio)
    rmse_scaled = np.mean(rmse.T, axis=0).reshape(1, -1)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0).reshape(1, -1)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # Combine all features into a single array
    features = np.concatenate([zcr_scaled, spectral_centroid_scaled, spectral_bandwidth_scaled,
                               chroma_stft_scaled, rmse_scaled, spectral_contrast_scaled, mfccs_scaled], axis=1)

    return features

# Memuat model dan scaler
try:
    model = load_model('./saved_models/audio_classification.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Dictionary untuk menghubungkan label prediksi dengan nama instrumen
label_to_instrument = {
    1: "Sound Guitar",
    2: "Sound Drum",
    3: "Sound Violin",
    4: "Sound Piano"
}

# Fungsi untuk klasifikasi
def classify_audio(file):
    features = Feature_extractor(file)
    if features is None:
        return "Error in feature extraction"
    
    try:
        features = features.reshape(1, -1)
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        predicted_label = np.argmax(predictions, axis=-1)
        prediction_class = label_encoder.inverse_transform(predicted_label)[0]
        instrument_name = label_to_instrument.get(prediction_class, "Unknown Instrument")
        return instrument_name
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return "Error in classification"

# Aplikasi Streamlit
def app():
    st.header('Program Klasifikasi Musik', divider='rainbow')
    st.title("MSIB BATCH 6 BISA AI Academy")
    st.write("Uji coba inferensi NLP")

    uploaded_file = st.file_uploader("Upload a file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Menampilkan file audio
        st.audio(uploaded_file, format='audio/wav' if uploaded_file.type == 'audio/wav' else 'audio/mp3')
        st.write("File audio berhasil diunggah dan diputar.")

        # Klasifikasi dan menampilkan hasilnya
        hasil_klasifikasi = classify_audio(uploaded_file)
        st.subheader('Hasil Klasifikasi Menunjukkan')
        st.write(f"Data termasuk dalam kelas: {hasil_klasifikasi}")

    # Menambahkan elemen lainnya
    st.subheader("Informasi Tambahan")
    st.write("""
    - *Fitur-fitur yang digunakan:* Zero Crossing Rate, Spectrogram, dll.
    - *Model yang digunakan:* Contoh Model (misal: LSTM, SVM, dll.)
    - *Akurasi Model:* 95%
    """)

    # Menambahkan informasi hak cipta di bagian bawah
    st.markdown("---")
    st.markdown("© 2024 MSIB BATCH 6 BISA AI Academy")
    st.markdown("Program ini dikembangkan untuk tujuan pendidikan dalam rangkaian program MSIB BATCH 6.")

if __name__ == '__main__':
    app()
