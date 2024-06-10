import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from io import BytesIO

# Fungsi untuk ekstraksi fitur audio
def feature_extractor(file):
    # Membaca file audio dari BytesIO
    audio, sample_rate = librosa.load(BytesIO(file.read()), sr=None, res_type='kaiser_fast')

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)

    features = np.concatenate([zcr, spectral_centroid, spectral_bandwidth, chroma_stft, rmse, spectral_contrast, mfccs])

    return features

# Memuat model dan scaler
model = load_model('./saved_models/audio_classification.keras')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Dictionary untuk menghubungkan label prediksi dengan nama instrumen
label_to_instrument = {
    1: "Sound Guitar",
    2: "Sound Drum",
    3: "Sound Violin",
    4: "Sound Piano"
}

# Fungsi untuk klasifikasi
def classify_audio(file):
    features = feature_extractor(file).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    predicted_label = np.argmax(predictions, axis=-1)
    prediction_class = label_encoder.inverse_transform(predicted_label)[0]
    instrument_name = label_to_instrument.get(prediction_class, "Unknown Instrument")
    return instrument_name

# Aplikasi Streamlit
def app():
    st.title("Klasifikasi Instrumen Musik")
    st.write("Unggah file audio (.wav atau .mp3) untuk mengklasifikasikan jenis instrumen musiknya.")

    uploaded_file = st.file_uploader("Unggah file audio", type=['wav', 'mp3'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav' if uploaded_file.type == 'audio/wav' else 'audio/mp3')
        st.write("File audio berhasil diunggah dan diputar.")

        try:
            hasil_klasifikasi = classify_audio(uploaded_file)
            st.subheader('Hasil Klasifikasi Menunjukkan')
            st.write(f"Data termasuk dalam kelas: {hasil_klasifikasi}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    st.subheader("Informasi Tambahan")
    st.write("""
    - *Fitur-fitur yang digunakan:* Zero Crossing Rate, Spectrogram, dll.
    - *Model yang digunakan:* LSTM
    - *Akurasi Model:* 95%
    """)

    st.markdown("---")
    st.markdown("© 2024 MSIB BATCH 6 BISA AI Academy")
