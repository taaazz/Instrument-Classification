import streamlit as st

# Fungsi untuk halaman welcome
def welcome_page():
    st.title("Selamat Datang di Program Klasifikasi Musik")
    st.write("Ini adalah halaman selamat datang untuk program klasifikasi musik.")
    if st.button("Mulai"):
        st.session_state.page = 'main'
    st.write("Tekan 2 kali yaa")
# Fungsi untuk halaman utama
def main_page():
    st.header('Program Klasifikasi Musik', divider='rainbow')
    st.title("MSIB BATCH 6 BISA AI Academy")
    st.write("Uji coba inferensi NLP")

    uploaded_file = st.file_uploader("Upload a file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Menampilkan file audio
        st.audio(uploaded_file, format='audio/wav' if uploaded_file.type == 'audio/wav' else 'audio/mp3')
        st.write("File audio berhasil diunggah dan diputar.")

        # Kode untuk klasifikasi dan menampilkan hasilnya (dummy text sebagai contoh)
        # Misalnya, Anda dapat memanggil fungsi klasifikasi di sini dan menampilkan hasilnya
        # hasil_klasifikasi = fungsi_klasifikasi(uploaded_file)
        hasil_klasifikasi = "Contoh Kelas"  # Ganti dengan hasil klasifikasi yang sebenarnya
        st.subheader('Hasil Klasifikasi Menunjukkan')
        st.write(f"Data termasuk dalam kelas: {hasil_klasifikasi}")

    # Menambahkan elemen lainnya
    st.subheader("Informasi Tambahan")
    st.write("""
    - **Fitur-fitur yang digunakan:** Zero Crossing Rate, Spectrogram, dll.
    - **Model yang digunakan:** Contoh Model (misal: Random Forest, SVM, dll.)
    - **Akurasi Model:** 95%
    """)

    # Menambahkan informasi hak cipta di bagian bawah
    st.markdown("---")
    st.markdown("© 2024 MSIB BATCH 6 BISA AI Academy")
    st.markdown("Program ini dikembangkan untuk tujuan pendidikan dalam rangkaian program MSIB BATCH 6.")

    # Tombol kembali ke halaman welcome
    if st.button("Kembali"):
        st.session_state.page = 'welcome'
    st.write("Tekan 2 kali yaa")
# Menetapkan halaman awal
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Navigasi halaman
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'main':
    main_page()
