import plotly.graph_objects as go

def plot_comparison(opening_scores, qa_scores):
    """
    Membuat Grouped Bar Chart untuk membandingkan skor sentimen.
    
    Args:
        opening_scores (dict): Skor VADER untuk Opening Speech.
        qa_scores (dict): Skor VADER untuk Q&A Session.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    categories = ['Positif', 'Netral', 'Negatif']
    
    # Mengambil nilai pos, neu, neg
    opening_values = [opening_scores['pos'], opening_scores['neu'], opening_scores['neg']]
    qa_values = [qa_scores['pos'], qa_scores['neu'], qa_scores['neg']]
    
    fig = go.Figure(data=[
        go.Bar(name='Opening Speech', x=categories, y=opening_values, marker_color='#1f77b4'),
        go.Bar(name='Q&A Session', x=categories, y=qa_values, marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        title='Perbandingan Sentimen: Opening Speech vs Q&A Session',
        xaxis_title='Kategori Sentimen',
        yaxis_title='Skor Proporsi',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def plot_gauge(opening_compound, qa_compound):
    """
    Membuat Gauge Chart untuk membandingkan skor Compound.
    
    Args:
        opening_compound (float): Skor compound Opening.
        qa_compound (float): Skor compound Q&A.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = qa_compound,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Q&A Compound Score (vs Opening)"},
        delta = {'reference': opening_compound},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [-1, -0.05], 'color': "red"},
                {'range': [-0.05, 0.05], 'color': "gray"},
                {'range': [0.05, 1], 'color': "green"}],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': opening_compound}}))

    fig.update_layout(height=400)
    return fig

def plot_wordcloud(text):
    """
    Membuat Word Cloud dari teks dengan filter POS Tagging (Hanya Noun & Adjective).
    
    Args:
        text (str): Teks input.
        
    Returns:
        matplotlib.figure.Figure: Objek gambar Matplotlib.
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    
    # Download NLTK resources if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # 1. Tokenisasi & POS Tagging
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    
    # 2. Whitelist Filter: Hanya ambil Noun (NN*) dan Adjective (JJ*)
    # NN: Noun, singular; NNS: Noun, plural; JJ: Adjective; JJR: Adj, comparative; JJS: Adj, superlative
    allowed_tags = {'NN', 'NNS', 'JJ', 'JJR', 'JJS'}
    
    # 3. Custom Blacklist (Kata benda umum yang tidak informatif)
    custom_stopwords = set([
        'percent', 'year', 'month', 'today', 'term', 'point', 'number', 'million', 'billion',
        'thing', 'way', 'part', 'lot', 'bit', 'side', 'type', 'kind', 'sort', 'sense',
        'question', 'answer', 'mr', 'ms', 'chair', 'powell', 'chairman', 'operator',
        'meeting', 'committee', 'reserve', 'federal', 'fed', 'bank', 'system',
        'program', 'statement', 'guidance', 'tool', 'support', 'policy', 'rate', # Policy & Rate sering muncul, opsional dihapus jika terlalu dominan
        'time', 'period', 'level', 'range', 'goal', 'objective', 'mandate',
        'michelle', 'smith', 'hi', 'hello', 'thanks', 'thank', 'please',
        'quarter', 'half', 'basis', 'pace', 'outlook', 'projection', 'view',
        'participant', 'member', 'colleague', 'staff', 'governor', 'president'
    ])
    
    # Gabungkan dengan stopwords bawaan NLTK
    final_stopwords = set(stopwords.words('english')).union(custom_stopwords)
    
    # Filter token
    filtered_words = []
    for word, tag in tagged_tokens:
        if tag in allowed_tags and word.isalpha() and word not in final_stopwords and len(word) > 2:
            filtered_words.append(word)
            
    clean_text = " ".join(filtered_words)
    
    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(clean_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def plot_historical_trend(historical_data, current_date=None, current_score=None):
    """
    Membuat Line Chart tren sentimen historis.
    
    Args:
        historical_data (list): List of dict data historis.
        current_date (date, optional): Tanggal file yang sedang dianalisis.
        current_score (float, optional): Skor compound file yang sedang dianalisis.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    dates = [item['date'] for item in historical_data]
    scores = [item['compound'] for item in historical_data]
    
    fig = go.Figure()
    
    # Garis Tren Historis
    fig.add_trace(go.Scatter(
        x=dates, 
        y=scores, 
        mode='lines+markers', 
        name='Historical Trend', 
        line=dict(color='gray', width=2, dash='dot'),
        marker=dict(size=6, color='gray')
    ))
    
    # Marker untuk File Saat Ini (Jika ada)
    if current_date and current_score is not None:
        fig.add_trace(go.Scatter(
            x=[current_date],
            y=[current_score],
            mode='markers',
            name='Current File',
            marker=dict(color='blue', size=15, symbol='star'),
            text=[f"Current: {current_score:.4f}"],
            hoverinfo='text+x+y'
        ))
    
    fig.update_layout(
        title='Tren Sentimen Historis The Fed (2020-2025)',
        xaxis_title='Tanggal Pertemuan',
        yaxis_title='Skor Compound VADER',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_topic_sentiment(topic_scores):
    """
    Membuat Bar Chart sentimen per topik.
    
    Args:
        topic_scores (dict): Skor sentimen per topik.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    topics = list(topic_scores.keys())
    scores = list(topic_scores.values())
    
    colors = ['green' if s >= 0 else 'red' for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(x=topics, y=scores, marker_color=colors)
    ])
    
    return fig

def plot_sentiment_flow(opening_sentences, qa_sentences):
    """
    Membuat Line Chart untuk alur sentimen (Sentiment Flow).
    
    Args:
        opening_sentences (list): List of dict dari get_sentence_scores.
        qa_sentences (list): List of dict dari get_sentence_scores.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    import pandas as pd
    
    # Prepare DataFrames
    df_op = pd.DataFrame(opening_sentences)
    df_qa = pd.DataFrame(qa_sentences)
    
    # Add rolling average for smoother lines
    if not df_op.empty:
        df_op['rolling'] = df_op['compound'].rolling(window=5, min_periods=1).mean()
    if not df_qa.empty:
        df_qa['rolling'] = df_qa['compound'].rolling(window=5, min_periods=1).mean()
        
    fig = go.Figure()
    
    # Opening Line
    if not df_op.empty:
        fig.add_trace(go.Scatter(
            x=df_op['seq'], 
            y=df_op['rolling'], 
            mode='lines', 
            name='Opening Speech (Trend)',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_op['seq'], 
            y=df_op['compound'], 
            mode='markers', 
            name='Opening (Raw)',
            marker=dict(color='#1f77b4', size=4, opacity=0.3),
            showlegend=False
        ))
        
    # Q&A Line
    if not df_qa.empty:
        fig.add_trace(go.Scatter(
            x=df_qa['seq'], 
            y=df_qa['rolling'], 
            mode='lines', 
            name='Q&A Session (Trend)',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_qa['seq'], 
            y=df_qa['compound'], 
            mode='markers', 
            name='Q&A (Raw)',
            marker=dict(color='#ff7f0e', size=4, opacity=0.3),
            showlegend=False
        ))
        
    fig.update_layout(
        title='Alur Sentimen (Sentiment Flow)',
        xaxis_title='Urutan Kalimat',
        yaxis_title='Skor Sentimen (Rolling Avg)',
        yaxis=dict(range=[-1, 1]),
        template='plotly_white',
        hovermode="x unified",
        shapes=[
            # Green Zone (Optimistic)
            dict(
                type="rect", xref="paper", yref="y",
                x0=0, y0=0.05, x1=1, y1=1,
                fillcolor="green", opacity=0.05, layer="below", line_width=0,
            ),
            # Red Zone (Pessimistic)
            dict(
                type="rect", xref="paper", yref="y",
                x0=0, y0=-1, x1=1, y1=-0.05,
                fillcolor="red", opacity=0.05, layer="below", line_width=0,
            )
        ]
    )
    
    return fig

def highlight_text(text, sentiment_lexicon=None):
    """
    Menandai kata-kata sentimen dalam teks untuk Streamlit.
    
    Args:
        text (str): Kalimat input.
        sentiment_lexicon (dict): Dictionary kata dan skornya.
        
    Returns:
        str: Teks dengan format markdown warna.
    """
    if sentiment_lexicon is None:
        return text
        
    words = text.split()
    highlighted_words = []
    
    for word in words:
        # Simple cleaning for matching
        clean_word = word.lower().strip('.,!?"\'')
        
        if clean_word in sentiment_lexicon:
            score = sentiment_lexicon[clean_word]
            if score > 0:
                highlighted_words.append(f":green[{word}]")
            elif score < 0:
                highlighted_words.append(f":red[{word}]")
            else:
                highlighted_words.append(word)
        else:
            highlighted_words.append(word)
            
    return " ".join(highlighted_words)


def plot_keyword_trend(keyword_data, keyword):
    """
    Membuat Scatter Plot untuk tren sentimen kata kunci tertentu.
    
    Args:
        keyword_data (list): List of dict [{'seq': int, 'compound': float, 'text': str}, ...]
        keyword (str): Kata kunci yang dianalisis.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    if not keyword_data:
        return go.Figure()
        
    seqs = [item['seq'] for item in keyword_data]
    scores = [item['compound'] for item in keyword_data]
    texts = [item['text'] for item in keyword_data]
    
    # Determine colors based on score
    colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' for s in scores]
    
    fig = go.Figure()
    
    # Add Scatter trace
    fig.add_trace(go.Scatter(
        x=seqs,
        y=scores,
        mode='markers',
        name=f'"{keyword}" Context',
        marker=dict(size=10, color=colors, line=dict(width=1, color='black')),
        text=texts,
        hoverinfo='text+y'
    ))
    
    # Add zero line
    fig.add_shape(type="line",
        x0=min(seqs)-1, y0=0, x1=max(seqs)+1, y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    fig.update_layout(
        title=f'Distribusi Sentimen Konteks Kata: "{keyword}"',
        xaxis_title='Urutan Kalimat dalam Transkrip',
        yaxis_title='Skor Sentimen',
        yaxis=dict(range=[-1.1, 1.1]),
        template='plotly_white',
        showlegend=False
    )
    
    return fig
