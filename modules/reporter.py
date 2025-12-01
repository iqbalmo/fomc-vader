
import plotly.io as pio

def generate_html_report(filename, opening_scores, qa_scores, topic_scores, fig_comparison, fig_flow, highlights):
    """
    Generates a standalone HTML report with embedded Plotly charts.
    """
    
    # Convert Plotly figures to HTML divs
    # include_plotlyjs='cdn' ensures the file is smaller but requires internet to render charts.
    # If offline usage is critical, we can use 'include_plotlyjs=True' but file size increases (3MB+).
    # For now, CDN is standard for web apps.
    div_comparison = pio.to_html(fig_comparison, full_html=False, include_plotlyjs='cdn')
    div_flow = pio.to_html(fig_flow, full_html=False, include_plotlyjs=False) # JS already included by first chart
    
    # Calculate difference
    diff = qa_scores['compound'] - opening_scores['compound']
    diff_text = "Lebih Positif" if diff > 0 else "Lebih Negatif"
    diff_color = "green" if diff > 0 else "red"
    
    # Highlights HTML
    pos_highlights = ""
    if highlights['positive']:
        for item in highlights['positive']:
             pos_highlights += f"<li class='highlight-pos'><b>{item['source']}:</b> \"{item['text']}\"</li>"
    else:
        pos_highlights = "<li>Tidak ada kalimat yang sangat positif.</li>"
        
    neg_highlights = ""
    if highlights['negative']:
        for item in highlights['negative']:
             neg_highlights += f"<li class='highlight-neg'><b>{item['source']}:</b> \"{item['text']}\"</li>"
    else:
        neg_highlights = "<li>Tidak ada kalimat yang sangat negatif.</li>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Laporan Analisis Sentimen FOMC</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max_width: 1000px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; border-left: 5px solid #2980b9; padding-left: 10px; }}
            .metric-container {{ display: flex; justify-content: space-between; margin: 20px 0; background: #f8f9fa; padding: 20px; border-radius: 8px; }}
            .metric {{ text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            .diff-val {{ color: {diff_color}; }}
            .chart-container {{ margin: 20px 0; border: 1px solid #eee; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
            .highlight-pos {{ background-color: #e8f5e9; border-left: 4px solid #4caf50; }}
            .highlight-neg {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #95a5a6; border-top: 1px solid #eee; padding-top: 20px; }}
            @media print {{
                .chart-container {{ break-inside: avoid; }}
                body {{ padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <h1>Laporan Analisis Sentimen FOMC</h1>
        <p style="text-align: center;">File: <b>{filename}</b></p>
        
        <h2>1. Ringkasan Eksekutif</h2>
        <div class="metric-container">
            <div class="metric">
                <div class="metric-label">Opening Speech</div>
                <div class="metric-value">{opening_scores['compound']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Q&A Session</div>
                <div class="metric-value">{qa_scores['compound']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Selisih</div>
                <div class="metric-value diff-val">{diff:+.4f} ({diff_text})</div>
            </div>
        </div>
        
        <h2>2. Sorotan Penting (Key Highlights)</h2>
        <h3>Kalimat Paling Optimis</h3>
        <ul>{pos_highlights}</ul>
        
        <h3>Kalimat Paling Pesimis/Waspada</h3>
        <ul>{neg_highlights}</ul>
        
        <h2>3. Visualisasi Perbandingan</h2>
        <div class="chart-container">
            {div_comparison}
        </div>
        
        <h2>4. Alur Sentimen (Sentiment Flow)</h2>
        <div class="chart-container">
            {div_flow}
        </div>
        
        <div class="footer">
            <p>Dibuat dengan FOMC VADER Analyzer. Laporan ini digenerate secara otomatis.</p>
            <p><i>Disclaimer: Analisis ini menggunakan metode VADER dengan leksikon finansial khusus. Hasil analisis adalah indikasi, bukan nasihat investasi mutlak.</i></p>
        </div>
    </body>
    </html>
    """
    
    return html_content.encode('utf-8')

def generate_pdf_report(filename, opening_scores, qa_scores, topic_scores, fig_comparison, fig_flow, highlights, conclusion_data, certainty_op, certainty_qa):
    """
    Generates a PDF report by converting Plotly charts to images and using xhtml2pdf.
    """
    import base64
    from xhtml2pdf import pisa
    from io import BytesIO
    
    # 1. Convert Plotly Figures to Static Images (Base64)
    # Requires 'kaleido' package
    # Optimization: Reduced scale from 2 to 1 for faster generation
    img_bytes_comp = fig_comparison.to_image(format="png", width=800, height=400, scale=1)
    img_base64_comp = base64.b64encode(img_bytes_comp).decode('utf-8')
    
    img_bytes_flow = fig_flow.to_image(format="png", width=800, height=400, scale=1)
    img_base64_flow = base64.b64encode(img_bytes_flow).decode('utf-8')
    
    # 2. Prepare HTML Content for PDF (Simpler CSS than web version)
    diff = qa_scores['compound'] - opening_scores['compound']
    diff_text = "Lebih Positif" if diff > 0 else "Lebih Negatif"
    diff_color = "green" if diff > 0 else "red"
    
    pos_highlights = ""
    if highlights['positive']:
        for item in highlights['positive']:
             pos_highlights += f"<li class='highlight-pos'><b>{item['source']}:</b> \"{item['text']}\"</li>"
    else:
        pos_highlights = "<li>Tidak ada kalimat yang sangat positif.</li>"
        
    neg_highlights = ""
    if highlights['negative']:
        for item in highlights['negative']:
             neg_highlights += f"<li class='highlight-neg'><b>{item['source']}:</b> \"{item['text']}\"</li>"
    else:
        neg_highlights = "<li>Tidak ada kalimat yang sangat negatif.</li>"

    # Topic Rows
    topic_rows = ""
    for topic, score in topic_scores.items():
        topic_rows += f"<tr><td>{topic.capitalize()}</td><td>{score:.4f}</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica, sans-serif; line-height: 1.5; color: #333; }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 20px; border-left: 4px solid #2980b9; padding-left: 10px; }}
            h3 {{ color: #34495e; margin-top: 15px; }}
            .metric-table {{ width: 100%; margin: 20px 0; border-collapse: collapse; }}
            .metric-table td, .metric-table th {{ padding: 10px; text-align: center; border: 1px solid #ddd; }}
            .metric-value {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            .diff-val {{ color: {diff_color}; }}
            .chart-img {{ width: 100%; margin: 10px 0; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 8px; padding: 8px; background-color: #f9f9f9; }}
            .highlight-pos {{ border-left: 4px solid #4caf50; }}
            .highlight-neg {{ border-left: 4px solid #f44336; }}
            .narrative-box {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 5px solid {conclusion_data['color']}; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 10px; color: #95a5a6; }}
        </style>
    </head>
    <body>
        <h1>Laporan Analisis Sentimen FOMC</h1>
        <p style="text-align: center;">File: <b>{filename}</b></p>
        
        <h2>1. Ringkasan Eksekutif</h2>
        <div class="narrative-box">
            <p><b>Status: {conclusion_data['status']}</b></p>
            <p>{conclusion_data['narrative']}</p>
        </div>
        
        <table class="metric-table">
            <tr>
                <td>
                    <div class="metric-label">Opening Speech</div>
                    <div class="metric-value">{opening_scores['compound']:.4f}</div>
                </td>
                <td>
                    <div class="metric-label">Q&A Session</div>
                    <div class="metric-value">{qa_scores['compound']:.4f}</div>
                </td>
                <td>
                    <div class="metric-label">Selisih</div>
                    <div class="metric-value diff-val">{diff:+.4f}</div>
                </td>
            </tr>
        </table>
        
        <h2>2. Analisis Mendalam</h2>
        <h3>Tingkat Keyakinan (Certainty Index)</h3>
        <table class="metric-table">
            <tr>
                <th>Sesi</th>
                <th>Skor (0-1)</th>
                <th>Label</th>
            </tr>
            <tr>
                <td>Opening Speech</td>
                <td>{certainty_op['score']:.2f}</td>
                <td>{certainty_op['label']}</td>
            </tr>
            <tr>
                <td>Q&A Session</td>
                <td>{certainty_qa['score']:.2f}</td>
                <td>{certainty_qa['label']}</td>
            </tr>
        </table>
        
        <h3>Sentimen per Topik</h3>
        <table class="metric-table">
            <tr>
                <th>Topik</th>
                <th>Skor Sentimen</th>
            </tr>
            {topic_rows}
        </table>
        
        <h2>3. Sorotan Penting</h2>
        <h3>Kalimat Paling Optimis</h3>
        <ul>{pos_highlights}</ul>
        
        <h3>Kalimat Paling Pesimis</h3>
        <ul>{neg_highlights}</ul>
        
        <h2>4. Visualisasi Perbandingan</h2>
        <img class="chart-img" src="data:image/png;base64,{img_base64_comp}" />
        
        <h2>5. Alur Sentimen</h2>
        <img class="chart-img" src="data:image/png;base64,{img_base64_flow}" />
        
        <div class="footer">
            <p>Dibuat dengan FOMC VADER Analyzer.</p>
        </div>
    </body>
    </html>
    """
    
    # 3. Convert HTML to PDF
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
    
    if pisa_status.err:
        return None
        
    return pdf_file.getvalue()
