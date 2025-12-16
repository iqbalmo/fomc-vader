import streamlit as st
import pandas as pd
from modules import preprocessor, analyzer, visualizer
import re
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(
    page_title="Analisis Sentimen FOMC VADER",
    layout="wide"
)

# --- Caching Functions (Performance Optimization) ---
@st.cache_data
def get_historical_data():
    """Cache historical data analysis to avoid re-processing 40+ files."""
    return analyzer.analyze_historical_data("fomc-transcript")

@st.cache_data
def process_transcript_cached(text):
    """Cache transcript processing (splitting, filtering, cleaning)."""
    # 1. Split
    opening_raw, qa_raw = preprocessor.split_transcript(text)
    
    if opening_raw is None or qa_raw is None:
        return None, None, None
    
    # 2. Filter Q&A
    qa_filtered = preprocessor.filter_speaker(qa_raw, "CHAIR POWELL")
    
    # 3. Clean
    opening = preprocessor.clean_text(opening_raw)
    qa = preprocessor.clean_text(qa_filtered)
    
    # Gabungkan
    cleaned_text = opening + " " + qa
    
    return opening, qa, cleaned_text

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Main Application ---
def main():
    st.title("Evaluasi Metode Domain-Adapted VADER untuk Analisis Dinamika Sentimen pada Konferensi Pers FOMC")
    st.markdown("""
    Aplikasi ini menganalisis sentimen dari transkrip pidato Ketua The Fed (Jerome Powell) 
    menggunakan metode **VADER (Valence Aware Dictionary and sEntiment Reasoner)**.
    Program akan membandingkan tonalitas antara **Pidato Pembuka** dan **Sesi Tanya Jawab**.
    """)
    
    # --- Sidebar Controls ---
    st.sidebar.header("📂 Input & Pengaturan")
    
    # File Uploader di Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Transkrip (File .txt)", type=['txt'])
    
    st.sidebar.divider()
    st.sidebar.header("📅 Data Historis")
    load_history = st.sidebar.checkbox("Tampilkan Data Historis", value=True)
    
    # --- Main Tabs ---
    tab_analysis, tab_history, tab_validation = st.tabs(["📊 Analisis File Upload", "📈 Analisis Tren Historis", "🧪 Validasi Ilmiah"])
    
    # --- Tab 1: Analisis File ---
    with tab_analysis:
        if uploaded_file is not None:
            # Membaca file
            try:
                text = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
                return

            # Processing
            with st.spinner('Memproses transkrip...'):
                opening, qa, cleaned_text = process_transcript_cached(text)
                
                if opening is None:
                    st.error("Gagal memisahkan transkrip! Separator tidak ditemukan. Pastikan transkrip mengandung frasa kunci yang sesuai.")
                    return
            
            # Analisis Sentimen
            try:
                opening_scores = analyzer.get_vader_score(opening)
                qa_scores = analyzer.get_vader_score(qa)
                
                # Analisis Flow (Kalimat)
                opening_sentences = analyzer.get_sentence_scores(opening)
                qa_sentences = analyzer.get_sentence_scores(qa)
                
                # Analisis Certainty
                certainty_opening = analyzer.analyze_certainty(opening)
                certainty_qa = analyzer.analyze_certainty(qa)
                
                # Analisis Topik
                topic_scores = analyzer.analyze_topic_sentiment(cleaned_text)
                
                # Smart Conclusion
                conclusion = analyzer.generate_smart_conclusion(opening_scores['compound'], qa_scores['compound'])
                
                # Calculate Highlights
                highlights = analyzer.extract_key_highlights(opening, qa)
                
                # Notification (Only once per file)
                if 'last_processed_file' not in st.session_state:
                    st.session_state['last_processed_file'] = None
                    
                if st.session_state['last_processed_file'] != uploaded_file.name:
                    st.success("✅ Transkrip berhasil diproses dan dianalisis.")
                    st.session_state['last_processed_file'] = uploaded_file.name
                
                # --- Export Data Feature (Reverted to Top) ---
                # Prepare DataFrame
                df_op = pd.DataFrame(opening_sentences)
                df_op['Source'] = 'Opening Speech'
                df_qa = pd.DataFrame(qa_sentences)
                df_qa['Source'] = 'Q&A Session'
                
                if not df_op.empty and not df_qa.empty:
                    df_export = pd.concat([df_op, df_qa], ignore_index=True)
                    # Reorder columns
                    cols = ['Source', 'seq', 'text', 'compound']
                    df_export = df_export[cols]
                    
                    csv = convert_df_to_csv(df_export)
                    
                    from modules import reporter
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="Unduh Data (CSV)",
                            data=csv,
                            file_name=f"analisis_{uploaded_file.name.replace('.txt', '')}.csv",
                            mime='text/csv',
                        )
                    with col_dl2:
                        # Generate PDF on demand
                        if st.button("Buat Laporan PDF"):
                            status = st.status("Membuat Laporan PDF...", expanded=True)
                            try:
                                status.write("Mengkonversi grafik...")
                                pdf_report = reporter.generate_pdf_report(
                                    uploaded_file.name,
                                    opening_scores,
                                    qa_scores,
                                    topic_scores,
                                    visualizer.plot_comparison(opening_scores, qa_scores),
                                    visualizer.plot_sentiment_flow(opening_sentences, qa_sentences),
                                    highlights,
                                    conclusion,
                                    certainty_opening,
                                    certainty_qa
                                )
                                status.write("Menyusun PDF...")
                                if pdf_report:
                                    status.update(label="PDF Siap!", state="complete", expanded=False)
                                    st.download_button(
                                        label="Unduh PDF",
                                        data=pdf_report,
                                        file_name=f"laporan_{uploaded_file.name.replace('.txt', '')}.pdf",
                                        mime='application/pdf',
                                    )
                                else:
                                    status.update(label="Gagal", state="error")
                                    st.error("Gagal membuat PDF.")
                            except Exception as e:
                                status.update(label="Error", state="error")
                                st.error(f"Error: {e}")
                
                # Menampilkan Teks & Word Cloud
                with st.expander("Lihat Transkrip & Word Cloud"):
                    st.subheader("Word Cloud (Kata Kunci Dominan)")
                    fig_wc = visualizer.plot_wordcloud(cleaned_text)
                    st.pyplot(fig_wc)
                    
                    # --- NEW FEATURE: Keyword Sentiment Context ---
                    st.divider()
                    st.subheader("🔍 Analisis Konteks Kata Kunci")
                    
                    # Get top keywords
                    top_keywords = analyzer.get_top_keywords(cleaned_text, n=30)
                    
                    if top_keywords:
                        selected_keyword = st.selectbox(
                            "Pilih kata kunci untuk melihat konteks sentimennya:",
                            options=top_keywords,
                            index=0
                        )
                        
                        if selected_keyword:
                            # Analyze context
                            keyword_context = analyzer.analyze_keyword_context(cleaned_text, selected_keyword)
                            
                            if keyword_context:
                                # Metrics
                                avg_score = sum(item['compound'] for item in keyword_context) / len(keyword_context)
                                count = len(keyword_context)
                                
                                k_col1, k_col2 = st.columns(2)
                                k_col1.metric("Frekuensi Kemunculan", f"{count} kali")
                                k_col2.metric("Rata-rata Sentimen", f"{avg_score:.4f}", analyzer.get_sentiment_label(avg_score))
                                
                                # Plot Trend
                                st.plotly_chart(visualizer.plot_keyword_trend(keyword_context, selected_keyword), use_container_width=True)
                                
                                # Show Sentences
                                st.markdown(f"**Daftar Kalimat yang Mengandung '{selected_keyword}':**")
                                for item in keyword_context:
                                    # Color code based on sentiment
                                    color = "gray"
                                    if item['compound'] > 0.05: color = "green"
                                    elif item['compound'] < -0.05: color = "red"
                                    
                                    # Highlight keyword in text
                                    highlighted_text = item['text'].replace(selected_keyword, f"**{selected_keyword}**")
                                    highlighted_text = highlighted_text.replace(selected_keyword.title(), f"**{selected_keyword.title()}**")
                                    
                                    st.markdown(f":{color}-background[Seq {item['seq']}] {highlighted_text} (Score: {item['compound']:.2f})")
                            else:
                                st.info("Kata kunci tidak ditemukan dalam konteks kalimat penuh.")
                    else:
                        st.warning("Tidak cukup data untuk mengekstrak kata kunci.")
                    
                    st.divider()
                    # --- END NEW FEATURE ---

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Pidato Pembuka")
                        st.text_area("Opening", opening, height=300)
                    with col2:
                        st.subheader("Sesi Tanya Jawab")
                        st.text_area("Q&A", qa, height=300)
                
                # Metrik Utama
                st.divider()
                st.header("Hasil Analisis Sentimen")
                
                # Display Smart Conclusion
                if conclusion['color'] == 'green':
                    st.success(f"**Kesimpulan: {conclusion['status']}**\n\n{conclusion['narrative']}")
                elif conclusion['color'] == 'red':
                    st.error(f"**Kesimpulan: {conclusion['status']}**\n\n{conclusion['narrative']}")
                else:
                    st.info(f"**Kesimpulan: {conclusion['status']}**\n\n{conclusion['narrative']}")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Skor Komposit Opening", f"{opening_scores['compound']:.4f}", analyzer.get_sentiment_label(opening_scores['compound']))
                with col_m2:
                    st.metric("Skor Komposit Q&A", f"{qa_scores['compound']:.4f}", analyzer.get_sentiment_label(qa_scores['compound']))
                with col_m3:
                    diff = qa_scores['compound'] - opening_scores['compound']
                    st.metric("Selisih (Q&A - Opening)", f"{diff:.4f}", "Lebih Positif" if diff > 0 else "Lebih Negatif")
                    
                # Uji Validitas Statistik (T-Test)
                st.caption("---")
                stat_results = analyzer.perform_statistical_test(opening_sentences, qa_sentences)
                
                col_stat1, col_stat2 = st.columns([1, 2])
                with col_stat1:
                    st.metric(
                        "P-Value (Uji T)", 
                        f"{stat_results['p_value']:.4f}",
                        "Signifikan (< 0.05)" if stat_results['is_significant'] else "Tidak Signifikan",
                        delta_color="normal" if stat_results['is_significant'] else "off"
                    )
                with col_stat2:
                    st.info(f"**Interpretasi Statistik:** {stat_results['narrative']}")

                # Certainty Index
                st.caption("---")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric("Tingkat Kepastian (Opening)", f"{certainty_opening['score']:.2f}", certainty_opening['label'], delta_color="off")
                with col_c2:
                    st.metric("Tingkat Kepastian (Q&A)", f"{certainty_qa['score']:.2f}", certainty_qa['label'], delta_color="off")

                # Key Highlights
                st.divider()
                st.subheader("Sorotan Penting (Key Highlights)")
                
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.success("##### Kalimat Paling Optimis")
                    if highlights['positive']:
                        for item in highlights['positive']:
                            badge = f":blue-background[{item['source']}]" if item['source'] == 'Opening Speech' else f":orange-background[{item['source']}]"
                            txt = visualizer.highlight_text(item['text'], analyzer.FINANCIAL_LEXICON)
                            st.markdown(f"{badge} *\"{txt}\"*")
                    else:
                        st.write("Tidak ada kalimat yang sangat positif.")
                        
                with col_h2:
                    st.error("##### Kalimat Paling Pesimis")
                    if highlights['negative']:
                        for item in highlights['negative']:
                            badge = f":blue-background[{item['source']}]" if item['source'] == 'Opening Speech' else f":orange-background[{item['source']}]"
                            txt = visualizer.highlight_text(item['text'], analyzer.FINANCIAL_LEXICON)
                            st.markdown(f"{badge} *\"{txt}\"*")
                    else:
                        st.write("Tidak ada kalimat yang sangat negatif.")

                # Visualisasi
                st.divider()
                st.subheader("Visualisasi Perbandingan")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bar Chart", "Gauge Chart", "Sentimen per Topik", "Alur Sentimen", "🤖 AI Topic Discovery"])
                
                with tab1:
                    st.plotly_chart(visualizer.plot_comparison(opening_scores, qa_scores), use_container_width=True)
                with tab2:
                    st.plotly_chart(visualizer.plot_gauge(opening_scores['compound'], qa_scores['compound']), use_container_width=True)
                with tab3:
                    st.plotly_chart(visualizer.plot_topic_sentiment(topic_scores), use_container_width=True)
                with tab4:
                    st.plotly_chart(visualizer.plot_sentiment_flow(opening_sentences, qa_sentences), use_container_width=True)
                with tab5:
                    st.caption("Menggunakan Unsupervised Learning (K-Means) dengan Optimasi Silhouette Score (Auto-K).")
                    with st.spinner("Melakukan Clustering & Optimasi..."):
                        # Perform Optimized Clustering
                        # Returns tuple: (results, optimal_k, silhouette_score)
                        cluster_results, best_k, best_score = analyzer.perform_optimized_clustering(cleaned_text)
                        
                        if cluster_results:
                            st.success(f"Optimal Clusters: **{best_k}** (Silhouette Score: per {best_score:.4f})")
                            st.plotly_chart(visualizer.plot_cluster_sentiment(cluster_results), use_container_width=True)
                            
                            # Show details
                            with st.expander("Lihat Detail Cluster"):
                                for c in cluster_results:
                                    st.markdown(f"**Cluster {c['cluster_id']+1}:** {c['label']} (Avg Score: {c['avg_sentiment']:.4f})")
                        else:
                            st.warning("Data tidak cukup untuk melakukan clustering (butuh lebih banyak kalimat panjang).")
                    
                # Interpretasi
                st.divider()
                st.header("Interpretasi Hasil")
                
                tab_inv, tab_awam = st.tabs(["Untuk Investor", "Untuk Orang Awam"])
                with tab_inv:
                    st.markdown(f"**Opening:** {analyzer.interpret_sentiment(opening_scores['compound'], 'investor')}")
                    st.markdown(f"**Q&A:** {analyzer.interpret_sentiment(qa_scores['compound'], 'investor')}")
                with tab_awam:
                    st.markdown(f"**Opening:** {analyzer.interpret_sentiment(opening_scores['compound'], 'general')}")
                    st.markdown(f"**Q&A:** {analyzer.interpret_sentiment(qa_scores['compound'], 'general')}")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👈 Silakan upload file transkrip (.txt) di sidebar untuk memulai.")

    # --- Tab 2: Analisis Historis ---
    with tab_history:
        st.header("Analisis Tren Historis (2020-2025)")
        
        if load_history:
            with st.spinner("Memuat data historis..."):
                historical_data = get_historical_data()
                
                if historical_data:
                    current_date = None
                    current_score = None
                    
                    # Cek file saat ini untuk plotting
                    if uploaded_file is not None and 'cleaned_text' in locals():
                        match = re.search(r'(\d{8})', uploaded_file.name)
                        if match:
                            try:
                                current_date = datetime.strptime(match.group(1), '%Y%m%d').date()
                                current_score = analyzer.get_vader_score(cleaned_text)['compound']
                            except ValueError:
                                pass
                    
                    # 1. Historical Trend Plot
                    st.plotly_chart(visualizer.plot_historical_trend(historical_data, current_date, current_score), use_container_width=True)
                    st.success(f"Menampilkan data dari {len(historical_data)} pertemuan FOMC.")
                    
                    # 2. Market Correlation Analysis (S&P 500)
                    st.divider()
                    st.subheader("🔗 Korelasi dengan S&P 500")
                    st.caption("Menganalisis hubungan antara Sentimen Pidato dengan Perubahan Pasar (% Change Close-Open) pada hari yang sama (atau hari perdagangan berikutnya).")
                    
                    # Calculate Correlation
                    corr_result = analyzer.calculate_market_correlation(historical_data)
                    
                    col_corr1, col_corr2 = st.columns([1, 2])
                    with col_corr1:
                        st.metric("Korelasi Pearson", f"{corr_result['correlation']:.4f}", f"P-Value: {corr_result['p_value']:.4f}")
                    with col_corr2:
                        st.info(f"{corr_result['text']}")
                        
                    # Scatter Plot
                    st.plotly_chart(visualizer.plot_market_correlation(historical_data, corr_result['text']), use_container_width=True)
                    
                else:
                    st.warning("Tidak ada data historis ditemukan di folder 'fomc-transcript'.")
        else:
            st.write("Centang 'Tampilkan Data Historis' di sidebar untuk melihat tren.")

    # --- Tab 3: Scientific Validation ---
    with tab_validation:
        st.header("🧪 Validasi Ilmiah (Experimental)")
        st.markdown("""
        Validasi validitas skor sentimen VADER Anda dengan membandingkannya melawan model **State-of-the-Art (SOTA)** 
        berbasis Deep Learning khusus finansial: **FinBERT (ProsusAI)**.
        """)
        
        st.info("⚠️ **Catatan:** Proses ini membutuhkan download model (~440MB) pada penggunaan pertama dan mungkin memerlukan waktu.")
        
        if uploaded_file is not None and 'cleaned_text' in locals():
            if st.button("🚀 Jalankan Validasi Silang (Cross-Validation)"):
                try:
                    with st.spinner("Memuat Model FinBERT & Melakukan Validasi... (Harap tunggu)"):
                        # Lazy import to avoid loading heavy model at startup
                        from modules.validator import ScientificValidator
                        from nltk.tokenize import sent_tokenize
                        
                        # Initialize Validator
                        validator = ScientificValidator()
                        
                        # Prepare data for validation
                        # We need list of sentences and their VADER scores
                        sentences = sent_tokenize(cleaned_text)
                        
                        # Filter short sentences
                        valid_sents = []
                        vader_scores = []
                        
                        for s in sentences:
                            if len(s.split()) > 5:
                                valid_sents.append(s)
                                vader_scores.append(analyzer.get_vader_score(s)['compound'])
                                
                        if not valid_sents:
                            st.warning("Data kalimat valid tidak cukup.")
                        else:
                            # Run Validation
                            # Sample size 50 for balance between speed and accuracy
                            val_results = validator.validate_against_sota(valid_sents, vader_scores, sample_size=50)
                            
                            st.divider()
                            st.subheader("Hasil Validasi Statistik")
                            
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                st.metric(
                                    "Korelasi Pearson (r)", 
                                    f"{val_results['correlation']:.4f}",
                                    help="Mendekati 1.0 berarti VADER Anda sangat sejalan dengan FinBERT."
                                )
                            with col_v2:
                                st.metric(
                                    "P-Value", 
                                    f"{val_results['p_value']:.4f}", 
                                    "Signifikan" if val_results['p_value'] < 0.05 else "Tidak Signifikan",
                                    delta_color="normal" if val_results['p_value'] < 0.05 else "off"
                                )
                                
                            # Scatter Plot
                            st.plotly_chart(validator.plot_validation_scatter(val_results), use_container_width=True)
                            
                            # Interpretation
                            if val_results['correlation'] > 0.5:
                                st.success("✅ **Validasi Sukses:** Metode VADER modifikasi Anda menunjukkan hasil yang konsisten dengan FinBERT (SOTA).")
                            elif val_results['correlation'] > 0.3:
                                st.info("ℹ️ **Validasi Moderat:** Ada korelasi positif, namun terdapat beberapa perbedaan interpretasi antara VADER dan FinBERT.")
                            else:
                                st.warning("⚠️ **Validasi Lemah:** Hasil VADER cukup berbeda dengan FinBERT. Evaluasi kembali lexicon Anda.")
                            
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat validasi: {e}")
                    # st.exception(e) # Uncomment for debug
        else:
            st.warning("Silakan upload transkrip terlebih dahulu di sidebar.")

if __name__ == "__main__":
    main()
