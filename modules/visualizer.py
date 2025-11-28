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

def plot_historical_trend(historical_data):
    """
    Membuat Line Chart tren sentimen historis.
    
    Args:
        historical_data (list): List of dict data historis.
        
    Returns:
        plotly.graph_objects.Figure: Objek grafik Plotly.
    """
    dates = [item['date'] for item in historical_data]
    scores = [item['compound'] for item in historical_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers', name='Sentiment Score', line=dict(color='firebrick', width=2)))
    
    fig.update_layout(
        title='Tren Sentimen Historis The Fed (2020-2025)',
        xaxis_title='Tanggal Pertemuan',
        yaxis_title='Skor Compound VADER',
        template='plotly_white'
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
    
    fig.update_layout(
        title='Sentimen Berdasarkan Topik',
        xaxis_title='Topik',
        yaxis_title='Skor Sentimen',
        yaxis=dict(range=[-1, 1]),
        template='plotly_white'
    )
    
    return fig
