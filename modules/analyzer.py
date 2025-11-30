from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Custom Financial Lexicon
# Kata-kata ini memiliki bobot sentimen khusus dalam konteks ekonomi/The Fed
FINANCIAL_LEXICON = {
    'robust': 2.0,
    'strong': 1.5,
    'growth': 1.5,
    'stable': 1.5,
    'expansion': 1.5,
    'resilient': 1.5,
    'optimistic': 1.0,
    'solid': 1.5,
    'anchored': 1.0,
    'recalibration': 0.5,
    'inflation': -1.5,
    'hike': -1.0,
    'turmoil': -2.5,
    'volatility': -1.5,
    'recession': -3.0,
    'weak': -1.5,
    'slowdown': -1.5,
    'cooling': -0.5,
    'cooled': -0.5,
    'uncertainty': -1.0,
    'risk': -1.0,
    'downside': -1.5,
    'pressure': -1.0,
    'restrictive': -1.0,
    'tightening': -0.5,
    'unemployment': -2.0,
    'crisis': -3.0,
    'painful': -2.0,
    'restrictive': -1.0,
    'pandemic': -2.0,
    'confidence': 1.5,
    'remains': 0.5,
    'carefully': 0.5,
    'increases': -1.0,
    'increase': -1.0,
    'easing': 1.0,
    'eased': 1.0,
    'moderating': 1.0,
    'bottlenecks': -1.5,
    'transitory': -0.5,
    'elevated': -1.5,
    'soft': 1.0, # Soft landing
    'hard': -1.5, # Hard landing
    
    # Logic Tokens (spaCy Result)
    'economic_positive': 2.5,
    'economic_negative': -2.5,
    
    # Domain Adaptation: Neutralizing Directional Words
    # Kata-kata ini sering dianggap negatif oleh VADER default,
    # tapi dalam konteks The Fed ("lower inflation", "rate cut") sifatnya netral/positif.
    # Kita ubah jadi 0.0 agar sentimen bergantung pada kata benda (noun) atau konteksnya.
    'lower': 0.0,
    'low': 0.0,
    'cut': 0.0,
    'drop': 0.0,
    'decrease': 0.0,
    'declining': 0.0,
    
    # Memastikan indikator negatif tetap negatif
    'unemployment': -1.5,
    'inflation': -1.5
}

# Load spaCy model globally once
import spacy
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

def get_vader_score(text):
    """
    Menghitung skor sentimen VADER dengan Custom Lexicon Keuangan.
    
    Args:
        text (str): Teks input.
        
    Returns:
        dict: Dictionary berisi skor {'compound': float, 'pos': float, 'neu': float, 'neg': float}.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    analyzer.lexicon.update(FINANCIAL_LEXICON)
    
    # Smart Context Logic (spaCy)
    # Mengubah kalimat berdasarkan logika ekonomi sebelum masuk VADER
    text = apply_economic_logic(text)
    
    scores = analyzer.polarity_scores(text)
    return scores

def apply_economic_logic(text):
    """
    Menggunakan Dependency Parsing (spaCy) untuk menerapkan logika ekonomi.
    Contoh: "Inflation falls" -> "Inflation_Good"
    """
    doc = NLP(text)
    
    # Definisi Indikator
    bad_indicators = {'inflation', 'unemployment', 'cpi', 'pce', 'prices', 'price', 'cost', 'risk', 'uncertainty', 'volatility', 'pressure'}
    good_indicators = {'growth', 'gdp', 'employment', 'jobs', 'hiring', 'demand', 'spending', 'investment', 'activity', 'expansion', 'recovery'}
    
    # Definisi Arah (Lemmatized)
    up_verbs = {'rise', 'increase', 'grow', 'climb', 'jump', 'accelerate', 'surge', 'high', 'elevated', 'up', 'peak', 'skyrocket'}
    down_verbs = {'fall', 'drop', 'decline', 'decrease', 'slow', 'cool', 'moderate', 'ease', 'lower', 'low', 'down', 'weak', 'soft', 'weaken'}
    
    new_tokens = []
    
    for token in doc:
        lemma = token.lemma_.lower()
        
        # Cek apakah token ini adalah indikator ekonomi
        if lemma in bad_indicators or lemma in good_indicators:
            # Cari modifier atau verb yang terhubung (Head atau Children)
            # Sederhana: Cek tetangga atau head
            context_found = False
            
            # Cek Head (misal: inflation <- falls)
            head_lemma = token.head.lemma_.lower()
            
            # Cek Children (misal: lower -> inflation)
            children_lemmas = [child.lemma_.lower() for child in token.children]
            
            # Logika: Bad Indicator
            if lemma in bad_indicators:
                if head_lemma in down_verbs or any(c in down_verbs for c in children_lemmas):
                    new_tokens.append("economic_positive") # Inflation Down = Good
                    context_found = True
                elif head_lemma in up_verbs or any(c in up_verbs for c in children_lemmas):
                    new_tokens.append("economic_negative") # Inflation Up = Bad
                    context_found = True
                    
            # Logika: Good Indicator
            elif lemma in good_indicators:
                if head_lemma in up_verbs or any(c in up_verbs for c in children_lemmas):
                    new_tokens.append("economic_positive") # Growth Up = Good
                    context_found = True
                elif head_lemma in down_verbs or any(c in down_verbs for c in children_lemmas):
                    new_tokens.append("economic_negative") # Growth Down = Bad
                    context_found = True
            
            if not context_found:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
            
    return " ".join(new_tokens)

def get_sentiment_label(compound_score):
    """
    Mendapatkan label sentimen berdasarkan skor compound.
    """
    if compound_score >= 0.05:
        return "Positif"
    elif compound_score <= -0.05:
        return "Negatif"
    else:
        return "Netral"

def interpret_sentiment(compound_score, audience='general'):
    """
    Memberikan interpretasi naratif berdasarkan skor dan audiens.
    
    Args:
        compound_score (float): Skor compound VADER.
        audience (str): 'investor' atau 'general'.
        
    Returns:
        str: Penjelasan naratif.
    """
    if audience == 'investor':
        if compound_score >= 0.05:
            return "Sinyal **Dovish/Optimis**. The Fed tampaknya percaya diri dengan pertumbuhan ekonomi. Ini bisa menjadi sinyal positif untuk pasar saham (Risk-On), namun perhatikan jika optimisme ini berkaitan dengan inflasi yang terkendali."
        elif compound_score <= -0.05:
            return "Sinyal **Hawkish/Pesimis**. Nada negatif mengindikasikan kekhawatiran terhadap inflasi atau risiko ekonomi. Pasar mungkin bereaksi negatif (Risk-Off) atau mengantisipasi kenaikan suku bunga lebih lanjut/ketidakpastian."
        else:
            return "Sinyal **Netral**. Tidak ada indikasi kuat ke arah Hawkish atau Dovish. Pasar mungkin akan *wait and see* menunggu data ekonomi selanjutnya."
            
    elif audience == 'general':
        if compound_score >= 0.05:
            return "Kabar **Baik**. Ketua The Fed menyampaikan pesan yang menenangkan. Ini berarti ekonomi dianggap kuat, lapangan kerja aman, dan harga-harga diharapkan stabil. Masa depan ekonomi terlihat cerah."
        elif compound_score <= -0.05:
            return "Kabar **Kurang Baik**. Ada kekhawatiran mengenai kondisi ekonomi, seperti harga barang yang naik (inflasi) atau pertumbuhan yang melambat. Kita mungkin perlu lebih berhati-hati dalam pengeluaran."
        else:
            return "Kabar **Biasa Saja**. Situasi ekonomi dianggap stabil, tidak ada gejolak besar yang perlu dikhawatirkan saat ini."
    
    return ""

def analyze_topic_sentiment(text):
    """
    Menganalisis sentimen berdasarkan topik spesifik.
    
    Args:
        text (str): Teks input.
        
    Returns:
        dict: Skor sentimen per topik.
    """
    topics = {
        'Inflation': ['inflation', 'price', 'cpi', 'pce', 'cost', 'expensive'],
        'Labor Market': ['labor', 'job', 'employment', 'unemployment', 'wage', 'hiring', 'worker'],
        'Growth': ['growth', 'gdp', 'economy', 'spending', 'investment', 'activity', 'expansion']
    }
    
    results = {}
    sentences = text.split('.')
    
    for topic, keywords in topics.items():
        topic_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                topic_sentences.append(sentence)
        
        if topic_sentences:
            full_topic_text = ". ".join(topic_sentences)
            score = get_vader_score(full_topic_text)
            results[topic] = score['compound']
        else:
            results[topic] = 0.0
            
    return results

def analyze_historical_data(directory):
    """
    Menganalisis tren sentimen historis dari file transkrip.
    
    Args:
        directory (str): Path direktori transkrip.
        
    Returns:
        list: List of dict [{'date': date, 'compound': score}, ...]
    """
    import os
    import re
    from datetime import datetime
    
    historical_data = []
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    for filename in files:
        # Extract date from filename (e.g., FOMCpresconf20200916.txt)
        match = re.search(r'(\d{8})', filename)
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Gunakan preprocessor sederhana di sini atau import jika perlu
                # Kita asumsikan text bersih cukup untuk VADER
                score = get_vader_score(text)
                
                historical_data.append({
                    'date': date_obj,
                    'compound': score['compound'],
                    'filename': filename
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    # Sort by date
    historical_data.sort(key=lambda x: x['date'])
    return historical_data

def extract_key_highlights(opening_text, qa_text, num=3):
    """
    Mengekstrak kalimat-kalimat dengan sentimen paling positif dan negatif,
    lengkap dengan label sumbernya (Opening vs Q&A).
    
    Args:
        opening_text (str): Teks Opening Speech.
        qa_text (str): Teks Q&A Session.
        num (int): Jumlah kalimat per kategori.
        
    Returns:
        dict: {'positive': [{'text': str, 'source': str}, ...], 'negative': [...]}
    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    from nltk.tokenize import sent_tokenize
    
    scored_sentences = []
    
    # Process Opening
    for sent in sent_tokenize(opening_text):
        if len(sent.split()) < 5: continue
        score = get_vader_score(sent)
        scored_sentences.append({'text': sent, 'score': score['compound'], 'source': 'Opening Speech'})
        
    # Process Q&A
    for sent in sent_tokenize(qa_text):
        if len(sent.split()) < 5: continue
        score = get_vader_score(sent)
        scored_sentences.append({'text': sent, 'score': score['compound'], 'source': 'Q&A Session'})
        
    # Sort by score
    scored_sentences.sort(key=lambda x: x['score'], reverse=True)
    
    # Top Positive
    top_positive = [item for item in scored_sentences if item['score'] > 0.05][:num]
    
    # Top Negative (Bottom of the list)
    top_negative = [item for item in scored_sentences if item['score'] < -0.05][-num:]
    # Reverse negative list to show most negative first (optional, but usually better)
    top_negative.sort(key=lambda x: x['score']) 
    
    return {
        'positive': top_positive,
        'negative': top_negative
    }

def get_sentence_scores(text):
    """
    Menghitung skor sentimen untuk setiap kalimat dalam teks.
    Berguna untuk visualisasi alur sentimen (Sentiment Flow).
    
    Args:
        text (str): Teks input.
        
    Returns:
        list: List of dict [{'text': str, 'compound': float, 'seq': int}, ...]
    """
    import nltk
    from nltk.tokenize import sent_tokenize
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    sentences = sent_tokenize(text)
    results = []
    
    for i, sent in enumerate(sentences):
        if len(sent.split()) < 3: continue # Skip kalimat terlalu pendek
        score = get_vader_score(sent)
        results.append({
            'seq': i + 1,
            'text': sent,
            'compound': score['compound']
        })
        
    return results

def analyze_certainty(text):
    """
    Menganalisis tingkat kepastian (Certainty Index) berdasarkan penggunaan modal verbs.
    
    Args:
        text (str): Teks input.
        
    Returns:
        dict: {'score': float, 'label': str}
        Score range: 0.0 (Sangat Tidak Pasti) - 1.0 (Sangat Pasti)
    """
    doc = NLP(text.lower())
    
    # Modal Verbs Categorization
    certainty_words = {'will', 'must', 'shall', 'definitely', 'certainly', 'clearly', 'undoubtedly', 'always', 'never'}
    uncertainty_words = {'may', 'might', 'could', 'possibly', 'probably', 'perhaps', 'unlikely', 'likely', 'seems', 'appears'}
    
    certain_count = 0
    uncertain_count = 0
    total_words = 0
    
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.text in certainty_words:
                certain_count += 1
            elif token.text in uncertainty_words:
                uncertain_count += 1
                
    if total_words == 0:
        return {'score': 0.5, 'label': 'Netral'}
        
    # Simple Ratio Metric
    # Semakin banyak uncertain words, score makin rendah.
    # Base score 0.5. 
    # Tiap certain word nambah score, tiap uncertain ngurangin.
    
    # Normalisasi sederhana
    score = 0.5 + (certain_count * 0.05) - (uncertain_count * 0.05)
    score = max(0.0, min(1.0, score)) # Clip between 0 and 1
    
    if score > 0.6:
        label = "Tegas / Pasti"
    elif score < 0.4:
        label = "Hati-hati / Tidak Pasti"
    else:
        label = "Netral / Terukur"
        
    return {
        'score': score,
        'label': label,
        'details': {'certain_count': certain_count, 'uncertain_count': uncertain_count}
    }

