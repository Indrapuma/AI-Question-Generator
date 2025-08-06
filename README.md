# EduRAG - Educational Question Generator

EduRAG adalah aplikasi berbasis Streamlit yang memungkinkan Anda membuat kuis edukasi secara otomatis dari dokumen PDF atau teks menggunakan teknologi Retrieval-Augmented Generation (RAG) dan AI (Google Gemini atau model lokal FLAN-T5).

## Fitur

- **Upload Dokumen**: Tambahkan materi edukasi dari file PDF atau teks.
- **Knowledge Base**: Sistem akan mengindeks dan menganalisis dokumen Anda.
- **AI Model**: Pilih antara Google Gemini (butuh API key) atau model lokal.
- **Generate Questions**: Buat pertanyaan (MCQ & Short Answer) berdasarkan topik dan tingkat kesulitan.
- **Visualisasi**: Lihat distribusi sumber dan topik dokumen.
- **Review**: Tinjau pertanyaan dan jawabannya, lengkap dengan penjelasan.

## Cara Menjalankan

1. **Install dependencies**  
   Pastikan Python 3.10+ sudah terpasang.  
   Install requirements:
   ```sh
   pip install -r requirements.txt
   ```

2. **Jalankan aplikasi**
   ```sh
   streamlit run main.py
   ```

3. **Akses aplikasi**  
   Buka browser ke alamat yang diberikan oleh Streamlit (biasanya http://localhost:8501).

## Konfigurasi Model

- **Google Gemini Pro**:  
  Aktifkan di sidebar dan masukkan API key dari [Google AI Studio](https://makersuite.google.com).
- **Model Lokal**:  
  Secara default menggunakan FLAN-T5 (akan otomatis diunduh saat pertama kali dijalankan).

## Struktur Proyek

- `main.py` — Streamlit app utama.
- `rag_engine.py` — Logika RAG, embedding, dan question generation.
- `requirements.txt` — Daftar dependensi (buat file ini jika belum ada).

## Catatan

- Semua pemrosesan dokumen dilakukan secara lokal, kecuali jika menggunakan Gemini Pro.
- API key hanya disimpan di sesi browser Anda, tidak dikirim ke server manapun.

## Lisensi

MIT License

---

**Dibuat dengan ❤️ untuk edukasi.**