# Analisis Sentimen FOMC VADER 📊🇺🇸

Aplikasi analisis sentimen berbasis Python untuk menganalisis transkrip konferensi pers **Federal Open Market Committee (FOMC)**. Proyek ini dirancang untuk membantu investor dan peneliti memahami *tonalitas* (Hawkish vs Dovish) dari pidato Ketua The Fed, Jerome Powell.

## 🚀 Fitur Utama

*   **Analisis Sentimen VADER yang Dikalibrasi**: Menggunakan algoritma VADER yang telah disesuaikan dengan *Custom Financial Lexicon* untuk akurasi tinggi pada teks ekonomi.
*   **Smart Context Logic (spaCy)**: Menggunakan NLP canggih (*Dependency Parsing*) untuk memahami konteks ekonomi (misal: *"Inflation falls"* = Positif, *"Growth slows"* = Negatif).
*   **Interpretasi Multi-Audiens**: Penjelasan hasil analisis dalam dua bahasa:
    *   **Investor**: Istilah pasar (Risk-On/Off, Hawkish/Dovish).
    *   **Publik**: Penjelasan dampak ekonomi sehari-hari.
*   **Sorotan Penting (Key Highlights)**: Ekstraksi otomatis kalimat paling optimis dan pesimis, lengkap dengan label sumber (Opening Speech vs Q&A).
*   **Visualisasi Interaktif**:
    *   **Historical Trend**: Grafik tren sentimen dari 2020-2025.
    *   **Word Cloud**: Visualisasi kata kunci dominan dengan filter cerdas (POS Tagging).
    *   **Topic Sentiment**: Analisis spesifik per topik (Inflasi, Tenaga Kerja, Pertumbuhan).

## 🛠️ Instalasi

1.  **Clone Repository**
    ```bash
    git clone https://github.com/username/analisis-sentimen.git
    cd analisis-sentimen/fomc-vader
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Program akan otomatis mendownload model spaCy (`en_core_web_sm`) dan data NLTK saat pertama kali dijalankan.*

## 💻 Cara Penggunaan

Jalankan aplikasi menggunakan Streamlit:

```bash
python -m streamlit run app.py
```

Aplikasi akan terbuka di browser (biasanya di `http://localhost:8501`).

## 📂 Struktur Proyek

```
fomc-vader/
├── app.py                  # Main Entry Point (Streamlit UI)
├── requirements.txt        # Daftar Library Python
├── modules/                # Paket Modul Logika
│   ├── preprocessor.py     # Pembersihan & Pemisahan Teks
│   ├── analyzer.py         # Logika Sentimen (VADER + spaCy)
│   └── visualizer.py       # Visualisasi (Plotly + WordCloud)
└── fomc-transcript/        # Dataset Transkrip (.txt)
```

## 🧠 Metodologi

Proyek ini menggabungkan pendekatan **Lexicon-Based** (VADER) dengan **Rule-Based NLP** (spaCy).
1.  **Preprocessing**: Pemisahan Opening Speech dan Q&A Session.
2.  **Smart Context**: Mengubah frasa ekonomi menjadi token sentimen tunggal (misal: `lower inflation` -> `economic_positive`).
3.  **Scoring**: Menghitung skor *Compound* VADER.
4.  **Interpretation**: Mengkonversi skor menjadi narasi yang mudah dipahami.

---
*Dibuat untuk Skripsi Analisis Sentimen Tonalitas The Fed.*
