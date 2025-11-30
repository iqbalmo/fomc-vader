import streamlit as st
from modules import preprocessor, analyzer, visualizer

# Konfigurasi Halaman
st.set_page_config(
    page_title="Analisis Sentimen FOMC VADER",
    layout="wide"
)

def main():
    st.title("Analisis Sentimen Perbandingan Tonalitas Pidato Ketua The Fed")
    st.markdown("""
    Aplikasi ini menganalisis sentimen dari transkrip pidato Ketua The Fed (Jerome Powell) 
    menggunakan metode **VADER (Valence Aware Dictionary and sEntiment Reasoner)**.
    Program akan membandingkan tonalitas antara **Pidato Pembuka** dan **Sesi Tanya Jawab**.
    """)
    
    # Tab Utama
    tab_analysis, tab_history = st.tabs(["📂 Analisis File Upload", "📅 Analisis Tren Historis"])
    
    with tab_analysis:
        # File Uploader
        uploaded_file = st.file_uploader("Upload Transkrip (File .txt)", type=['txt'])
        
        if uploaded_file is not None:
            # Membaca file
            try:
                text = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
                return

            # Preprocessing & Splitting
            with st.spinner('Memproses transkrip...'):
                # 1. Split dulu (masih ada tag <NAME>)
                opening_raw, qa_raw = preprocessor.split_transcript(text)
                
                if opening_raw is None or qa_raw is None:
                    st.error("Gagal memisahkan transkrip! Separator tidak ditemukan. Pastikan transkrip mengandung salah satu frasa kunci: 'I look forward to your questions', 'glad to take your questions', atau 'questions, please'.")
                    return
                
                # 2. Filter Q&A (Hanya ambil suara CHAIR POWELL)
                qa_filtered = preprocessor.filter_speaker(qa_raw, "CHAIR POWELL")
                
                # 3. Bersihkan tag dan spasi
                opening = preprocessor.clean_text(opening_raw)
                qa = preprocessor.clean_text(qa_filtered)
                
                # Gabungkan untuk wordcloud
                cleaned_text = opening + " " + qa
            
            st.success("Transkrip berhasil dipisahkan dan difilter (Hanya suara Chair Powell)!")
            

            
            # Analisis Sentimen
            try:
                opening_scores = analyzer.get_vader_score(opening)
                qa_scores = analyzer.get_vader_score(qa)
                
                # Analisis Flow Sentimen (Kalimat per Kalimat)
                opening_sentences = analyzer.get_sentence_scores(opening)
                qa_sentences = analyzer.get_sentence_scores(qa)
                
                # Analisis Certainty (Kepastian)
                certainty_opening = analyzer.analyze_certainty(opening)
                certainty_qa = analyzer.analyze_certainty(qa)
                
                # Analisis Topik
                topic_scores = analyzer.analyze_topic_sentiment(cleaned_text)
                
                # Menampilkan Teks Asli & Word Cloud
                with st.expander("Lihat Transkrip & Word Cloud"):
                    st.subheader("Word Cloud (Kata Kunci Dominan)")
                    fig_wc = visualizer.plot_wordcloud(cleaned_text)
                    st.pyplot(fig_wc)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Pidato Pembuka")
                        st.text_area("Opening", opening, height=300)
                    with col2:
                        st.subheader("Sesi Tanya Jawab")
                        st.text_area("Q&A", qa, height=300)
                
                # Menampilkan Metrik Utama
                st.divider()
                st.header("Hasil Analisis Sentimen")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric(
                        label="Opening Compound Score", 
                        value=f"{opening_scores['compound']:.4f}",
                        delta=analyzer.get_sentiment_label(opening_scores['compound'])
                    )
                    
                with col_m2:
                    st.metric(
                        label="Q&A Compound Score", 
                        value=f"{qa_scores['compound']:.4f}",
                        delta=analyzer.get_sentiment_label(qa_scores['compound'])
                    )
                    
                with col_m3:
                    diff = qa_scores['compound'] - opening_scores['compound']
                    st.metric(
                        label="Selisih (Q&A - Opening)", 
                        value=f"{diff:.4f}",
                        delta="Lebih Positif" if diff > 0 else "Lebih Negatif"
                    )
                    
                # Certainty Index
                st.caption("---")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric(
                        label="Tingkat Kepastian (Opening)",
                        value=f"{certainty_opening['score']:.2f}",
                        delta=certainty_opening['label'],
                        delta_color="off"
                    )
                with col_c2:
                    st.metric(
                        label="Tingkat Kepastian (Q&A)",
                        value=f"{certainty_qa['score']:.2f}",
                        delta=certainty_qa['label'],
                        delta_color="off"
                    )

                # Sorotan Penting (Key Highlights)
                st.divider()
                st.subheader("🔍 Sorotan Penting (Key Highlights)")
                st.caption("Kalimat-kalimat yang paling mempengaruhi skor sentimen.")
                
                highlights = analyzer.extract_key_highlights(opening, qa)
                
                col_h1, col_h2 = st.columns(2)
                
                with col_h1:
                    st.success("##### ✅ Kalimat Paling Optimis (Positif)")
                    if highlights['positive']:
                        for item in highlights['positive']:
                            source_badge = f":blue-background[{item['source']}]" if item['source'] == 'Opening Speech' else f":orange-background[{item['source']}]"
                            highlighted_text = visualizer.highlight_text(item['text'], analyzer.FINANCIAL_LEXICON)
                            st.markdown(f"{source_badge} *\"{highlighted_text}\"*")
                    else:
                        st.write("Tidak ada kalimat yang sangat positif.")
                        
                with col_h2:
                    st.error("##### ⚠️ Kalimat Paling Pesimis/Waspada (Negatif)")
                    if highlights['negative']:
                        for item in highlights['negative']:
                            source_badge = f":blue-background[{item['source']}]" if item['source'] == 'Opening Speech' else f":orange-background[{item['source']}]"
                            highlighted_text = visualizer.highlight_text(item['text'], analyzer.FINANCIAL_LEXICON)
                            st.markdown(f"{source_badge} *\"{highlighted_text}\"*")
                    else:
                        st.write("Tidak ada kalimat yang sangat negatif.")

                # Visualisasi
                st.divider()
                st.subheader("Visualisasi Perbandingan")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Gauge Chart", "Sentimen per Topik", "Alur Sentimen (Flow)"])
                
                with tab1:
                    fig_bar = visualizer.plot_comparison(opening_scores, qa_scores)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with tab2:
                    fig_gauge = visualizer.plot_gauge(opening_scores['compound'], qa_scores['compound'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with tab3:
                    fig_topic = visualizer.plot_topic_sentiment(topic_scores)
                    st.plotly_chart(fig_topic, use_container_width=True)
                    
                with tab4:
                    st.caption("Grafik ini menunjukkan perubahan sentimen kalimat demi kalimat sepanjang pidato.")
                    fig_flow = visualizer.plot_sentiment_flow(opening_sentences, qa_sentences)
                    st.plotly_chart(fig_flow, use_container_width=True)
                    
                # Interpretasi & Kesimpulan
                st.divider()
                st.header("Interpretasi Hasil")
                
                # Menentukan sentimen dominan untuk narasi umum
                if qa_scores['compound'] < opening_scores['compound']:
                    comparison_text = "Sesi Tanya Jawab terdengar **lebih negatif** (atau kurang optimis) dibandingkan Pidato Pembuka."
                elif qa_scores['compound'] > opening_scores['compound']:
                    comparison_text = "Sesi Tanya Jawab terdengar **lebih positif** dibandingkan Pidato Pembuka."
                else:
                    comparison_text = "Tonalitas antara Pidato Pembuka dan Tanya Jawab relatif **konsisten**."
                    
                st.subheader("Kesimpulan Perbandingan")
                st.info(comparison_text)
                
                st.subheader("Pandangan Mendalam")
                tab_investor, tab_awam = st.tabs(["📈 Untuk Investor", "👨‍👩‍👧‍👦 Untuk Orang Awam"])
                
                with tab_investor:
                    st.markdown("### Analisis Dampak Pasar")
                    st.markdown(f"**Pidato Pembuka:** {analyzer.interpret_sentiment(opening_scores['compound'], 'investor')}")
                    st.markdown(f"**Sesi Q&A:** {analyzer.interpret_sentiment(qa_scores['compound'], 'investor')}")
                    st.caption("*Catatan: Analisis ini menggunakan pembobotan kata khusus (Financial Lexicon) untuk mendeteksi sinyal Hawkish/Dovish.*")
                    
                with tab_awam:
                    st.markdown("### Apa Artinya Bagi Kita?")
                    st.markdown(f"**Awal Pidato:** {analyzer.interpret_sentiment(opening_scores['compound'], 'general')}")
                    st.markdown(f"**Saat Menjawab Pertanyaan:** {analyzer.interpret_sentiment(qa_scores['compound'], 'general')}")
                    st.caption("*Penjelasan sederhana mengenai dampak pidato terhadap kehidupan sehari-hari.*")
                
                # Disclaimer & Insight Box
                st.divider()
                st.warning("""
                **⚠️ Catatan Penting & Limitasi:**
                1.  **Filter Pembicara**: Analisis pada sesi Q&A telah difilter untuk **hanya mencakup jawaban Jerome Powell**. Pertanyaan wartawan (yang seringkali bernada skeptis/negatif) telah dihapus untuk akurasi yang lebih baik.
                2.  **Konteks Ekonomi**: Metode VADER berbasis leksikon (kamus kata). Meskipun telah dimodifikasi, ia mungkin masih kesulitan memahami konteks makroekonomi yang kompleks (misal: "Recession is unlikely" mungkin masih terdeteksi negatif karena kata "Recession").
                3.  **Sifat Alat**: Alat ini adalah pendukung keputusan, bukan nasihat investasi mutlak. Selalu gunakan pertimbangan profesional.
                """)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan analisis: {e}")
                import traceback
                st.code(traceback.format_exc())

    with tab_history:
        st.header("Analisis Tren Historis (2020-2025)")
        st.markdown("Grafik ini menunjukkan pergerakan sentimen The Fed dari waktu ke waktu berdasarkan arsip transkrip.")
        
        if st.button("Muat Data Historis"):
            with st.spinner("Menganalisis 40+ file transkrip..."):
                historical_data = analyzer.analyze_historical_data("fomc-transcript")
                
                if historical_data:
                    # Cek apakah ada file yang sedang dianalisis untuk ditandai
                    current_date = None
                    current_score = None
                    
                    if uploaded_file is not None and 'cleaned_text' in locals():
                        import re
                        from datetime import datetime
                        
                        # Coba ekstrak tanggal dari nama file (misal: FOMCpresconf20230503.txt)
                        match = re.search(r'(\d{8})', uploaded_file.name)
                        if match:
                            try:
                                current_date = datetime.strptime(match.group(1), '%Y%m%d').date()
                                # Hitung skor keseluruhan untuk konsistensi dengan data historis
                                overall_score = analyzer.get_vader_score(cleaned_text)
                                current_score = overall_score['compound']
                            except ValueError:
                                pass
                    
                    fig_trend = visualizer.plot_historical_trend(historical_data, current_date, current_score)
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    st.success(f"Berhasil menganalisis {len(historical_data)} dokumen pertemuan FOMC.")
                else:
                    st.error("Tidak ada data ditemukan di folder 'fomc-transcript'.")

if __name__ == "__main__":
    main()
