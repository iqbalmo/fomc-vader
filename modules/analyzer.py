from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats
import numpy as np

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
    'disinflation': 1.5,
    'disinflationary': 1.5,
    'normalization': 1.0,
    'headwinds': -1.5,
    'tailwinds': 1.5,
    'anchored': 1.5,
    'unanchored': -2.0,
    'vigilant': -0.5, # Often implies watching for bad things
    'data-dependent': 0.5, # Neutral/Cautious optimism
    'restrictive': -1.0,
    'tight': -1.0,
    'tightening': -1.0,
    'balanced': 1.0,
    'imbalance': -1.5,
    
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

# Hedge Words (Fed Speak) Damping Factors
# Kata-kata ini menandakan ketidakpastian atau "Fed Speak" yang halus.
# Jika ada kata ini, kita kurangi intesitas sentimen (damping factor).
HEDGE_MODIFIERS = {
    'likely': 0.8,
    'possibly': 0.7,
    'suggests': 0.8,
    'might': 0.7,
    'could': 0.7,
    'appears': 0.8,
    'seems': 0.8,
    'somewhat': 0.8,
    'relatively': 0.9,
    'essentially': 0.9,
    'may': 0.8,
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
    Menghitung skor sentimen VADER dengan Custom Lexicon Keuangan
    dan Logika Damping untuk Hedge Words (Fed Speak).
    
    Args:
        text (str): Teks input.
        
    Returns:
        dict: Dictionary berisi skor {'compound': float, 'pos': float, 'neu': float, 'neg': float}.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    analyzer.lexicon.update(FINANCIAL_LEXICON)
    
    # 1. Smart Context Logic (spaCy)
    # Mengubah kalimat berdasarkan logika ekonomi sebelum masuk VADER
    processed_text = apply_economic_logic(text)
    
    # 2. Basic VADER Score
    scores = analyzer.polarity_scores(processed_text)
    
    # 3. Hedge Words Damping Logic
    # Cek apakah ada hedge words di teks ASLI (sebelum diproses logic ekonomi jika perlu, 
    # tapi processed_text isinya token yang digabung, jadi cek raw text lebih aman untuk keyword matching)
    
    damping_factor = 1.0
    text_lower = text.lower()
    
    words = text_lower.split() # Simple split for check
    # Note: Ini simple check, ideally tokenized. Tapi cukup untuk keyword 'might', 'could' dll.
    
    for word, factor in HEDGE_MODIFIERS.items():
        if word in words:
            damping_factor *= factor
            
    # Apply damping to compound score only (biasakan intensitas berkurang)
    # Jika damping_factor < 1.0, skor mendekati 0.
    if damping_factor < 1.0:
        scores['compound'] = scores['compound'] * damping_factor
        
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
    Menganalisis tren sentimen historis dan menghubungkannya dengan data pasar (S&P 500).
    
    Args:
        directory (str): Path direktori transkrip.
        
    Returns:
        list: List of dict [{'date': date, 'compound': score, 'market_change': float}, ...]
    """
    import os
    import re
    from datetime import datetime, timedelta
    import yfinance as yf
    
    historical_data = []
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # Pre-fetch market data optimization?
    # Untuk simplicitas, kita fetch per tanggal tapi dengan range kecil.
    # Agar lebih cepat, kita bisa fetch all history sekali saja, tapi itu butuh range min-max date.
    # Kita pakai pendekatan per file dulu, jika lambat bisa dioptimasi nanti.
    
    print(f"Processing {len(files)} historical files...")
    
    for filename in files:
        # Extract date from filename (e.g., FOMCpresconf20200916.txt)
        match = re.search(r'(\d{8})', filename)
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                score = get_vader_score(text)
                
                # Fetch Market Data (S&P 500: ^GSPC)
                # Ambil data H sampai H+5 untuk handle weekend/holiday (ambil first trading day)
                start_date = date_obj
                end_date = date_obj + timedelta(days=5)
                
                ticker = yf.Ticker("^GSPC")
                hist = ticker.history(start=start_date, end=end_date)
                
                market_change = None
                
                if not hist.empty:
                    # Ambil hari pertama yang tersedia (bisa hari H atau besoknya jika libur)
                    row = hist.iloc[0]
                    # Hitung % Change: (Close - Open) / Open
                    market_change = ((row['Close'] - row['Open']) / row['Open']) * 100
                
                historical_data.append({
                    'date': date_obj,
                    'compound': score['compound'],
                    'market_change': market_change,
                    'filename': filename
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    # Sort by date
    historical_data.sort(key=lambda x: x['date'])
    return historical_data

def calculate_market_correlation(historical_data):
    """
    Menghitung korelasi Pearson antara Sentimen dan Perubahan Pasar.
    """
    from scipy import stats
    import numpy as np
    
    # Filter data valid (yang punya market_change)
    valid_data = [d for d in historical_data if d['market_change'] is not None]
    
    if len(valid_data) < 2:
        return {'correlation': 0.0, 'p_value': 1.0, 'narrative': 'Data tidak cukup.'}
        
    sentiments = [d['compound'] for d in valid_data]
    market_changes = [d['market_change'] for d in valid_data]
    
    corr_coef, p_value = stats.pearsonr(sentiments, market_changes)
    
    # Interpretasi
    if abs(corr_coef) < 0.3:
        strength = "Sangat Lemah"
    elif abs(corr_coef) < 0.5:
        strength = "Lemah"
    elif abs(corr_coef) < 0.7:
        strength = "Sedang"
    else:
        strength = "Kuat"
        
    direction = "Positif" if corr_coef > 0 else "Negatif"
    
    narrative = (
        f"Korelasi {direction} {strength} ({corr_coef:.4f}). "
        f"P-Value: {p_value:.4f}. "
    )
    
    if p_value < 0.05:
        narrative += "Hubungan ini Signifikan secara statistik."
    else:
        narrative += "Hubungan ini TIDAK Signifikan secara statistik (mungkin kebetulan)."
        
    return {
        'correlation': corr_coef,
        'p_value': p_value,
        'text': narrative
    }

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
    
    label = "Netral"
    if score > 0.6: label = "Tegas / Pasti"
    elif score < 0.4: label = "Hati-hati / Tidak Pasti"
    
    return {'score': score, 'label': label}

def generate_smart_conclusion(opening_score, qa_score):
    """
    Menghasilkan kesimpulan cerdas berdasarkan perbandingan skor.
    Threshold: 0.05
    """
    diff = qa_score - opening_score
    threshold = 0.05
    
    if diff > threshold:
        status = "Lebih Optimis"
        color = "green"
        narrative = (
            f"Sesi Tanya Jawab menunjukkan peningkatan sentimen positif sebesar {diff:.4f} dibandingkan Pidato Pembuka. "
            "Ini mengindikasikan bahwa Ketua The Fed memberikan klarifikasi yang menenangkan pasar atau "
            "menyampaikan pandangan yang lebih konstruktif saat merespons pertanyaan wartawan."
        )
    elif diff < -threshold:
        status = "Lebih Pesimis"
        color = "red"
        narrative = (
            f"Sesi Tanya Jawab menunjukkan penurunan sentimen sebesar {abs(diff):.4f} dibandingkan Pidato Pembuka. "
            "Hal ini menandakan adanya nada kehati-hatian atau peringatan risiko yang lebih kuat saat sesi diskusi, "
            "yang mungkin tidak terlalu ditekankan dalam naskah pidato resmi."
        )
    else:
        status = "Netral / Konsisten"
        color = "blue"
        narrative = (
            f"Tonalitas sentimen relatif konsisten antara Pidato Pembuka dan Sesi Tanya Jawab (selisih {diff:.4f} < 0.05). "
            "Ketua The Fed mempertahankan pesan yang stabil dan terukur, tanpa memberikan sinyal kejutan yang signifikan "
            "selama sesi diskusi."
        )
        
    return {
        'status': status,
        'color': color,
        'narrative': narrative
    }
    
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



def get_top_keywords(text, n=20):
    """
    Ekstrak kata-kata kunci paling sering muncul (selain stopwords).
    Menggunakan logika yang sama dengan WordCloud (POS Tagging: Noun & Adj).
    """
    from collections import Counter
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
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # 1. Tokenisasi & POS Tagging
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    
    # 2. Whitelist Filter: Hanya ambil Noun (NN*) dan Adjective (JJ*)
    allowed_tags = {'NN', 'NNS', 'JJ', 'JJR', 'JJS'}
    
    # 3. Custom Blacklist (Sama dengan visualizer.py)
    custom_stopwords = set([
        'percent', 'year', 'month', 'today', 'term', 'point', 'number', 'million', 'billion',
        'thing', 'way', 'part', 'lot', 'bit', 'side', 'type', 'kind', 'sort', 'sense',
        'question', 'answer', 'mr', 'ms', 'chair', 'powell', 'chairman', 'operator',
        'meeting', 'committee', 'reserve', 'federal', 'fed', 'bank', 'system',
        'program', 'statement', 'guidance', 'tool', 'support', 'policy', 'rate', 
        'time', 'period', 'level', 'range', 'goal', 'objective', 'mandate',
        'michelle', 'smith', 'hi', 'hello', 'thanks', 'thank', 'please',
        'quarter', 'half', 'basis', 'pace', 'outlook', 'projection', 'view',
        'participant', 'member', 'colleague', 'staff', 'governor', 'president',
        'morning', 'afternoon', 'evening', 'everyone', 'everybody', 'people',
        'think', 'going', 'see', 'say', 'know', 'look', 'come', 'make', 'take', 'get' # Common verbs often tagged as nouns in some contexts
    ])
    
    # Gabungkan dengan stopwords bawaan NLTK
    final_stopwords = set(stopwords.words('english')).union(custom_stopwords)
    
    # Important Economic Terms to KEEP (Remove from stopwords if present)
    # Meskipun ada di custom_stopwords visualizer (misal 'rate'), user mungkin ingin melihatnya di sini.
    # Tapi user minta "sesuaikan dengan filter wordcloud", jadi saya akan ikuti visualizer.py dulu.
    # Namun, 'inflation', 'jobs', 'growth' tidak ada di blacklist visualizer, jadi aman.
    
    # Filter token
    meaningful_words = []
    for word, tag in tagged_tokens:
        if tag in allowed_tags and word.isalpha() and word not in final_stopwords and len(word) > 2:
            meaningful_words.append(word)
    
    # Count
    counter = Counter(meaningful_words)
    return [word for word, count in counter.most_common(n)]

def analyze_keyword_context(text, keyword):
    """
    Menganalisis sentimen kalimat-kalimat yang mengandung keyword tertentu.
    """
    import nltk
    from nltk.tokenize import sent_tokenize
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    sentences = sent_tokenize(text)
    results = []
    
    keyword = keyword.lower()
    
    for i, sent in enumerate(sentences):
        if keyword in sent.lower():
            score = get_vader_score(sent)
            results.append({
                'seq': i + 1,
                'text': sent,
                'compound': score['compound']
            })
            
    return results

def perform_statistical_test(opening_sentences, qa_sentences):
    """
    Melakukan uji statistik (Independent T-Test) untuk membandingkan
    rata-rata sentimen antara Opening Speech dan Q&A Session.
    
    Args:
        opening_sentences (list): List of dict hasil get_sentence_scores untuk Opening.
        qa_sentences (list): List of dict hasil get_sentence_scores untuk Q&A.
        
    Returns:
        dict: {
            't_stat': float,
            'p_value': float,
            'is_significant': bool,
            'narrative': str
        }
    """
    # Extract compound scores
    opening_scores = [item['compound'] for item in opening_sentences]
    qa_scores = [item['compound'] for item in qa_sentences]
    
    # Calculate means (for narrative)
    mean_opening = np.mean(opening_scores) if opening_scores else 0
    mean_qa = np.mean(qa_scores) if qa_scores else 0
    
    # Check sufficiency of data
    if len(opening_scores) < 2 or len(qa_scores) < 2:
        return {
            't_stat': 0.0,
            'p_value': 1.0,
            'is_significant': False,
            'narrative': "Data tidak cukup untuk melakukan uji statistik yang valid."
        }
        
    # Perform Independent T-Test (Two-sided)
    # equal_var=False (Welch's t-test) karena varians mungkin berbeda
    t_stat, p_value = stats.ttest_ind(opening_scores, qa_scores, equal_var=False)
    
    is_significant = p_value < 0.05
    
    if is_significant:
        significance_text = "SIGNIFIKAN secara statistik"
        if mean_qa > mean_opening:
            direction = "lebih positif"
        else:
            direction = "lebih negatif"
        narrative = (
            f"Terdapat perbedaan yang {significance_text} (p-value: {p_value:.4f} < 0.05) "
            f"antara sentimen Opening dan Q&A. "
            f"Sesi Q&A secara signifikan {direction} dibandingkan Opening."
        )
    else:
        significance_text = "TIDAK SIGNIFIKAN secara statistik"
        narrative = (
            f"Perbedaan sentimen antara Opening dan Q&A {significance_text} (p-value: {p_value:.4f} >= 0.05). "
            "Setiap perbedaan rata-rata yang terlihat kemungkinan hanya kebetulan (variasi acak)."
        )
        
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'narrative': narrative
    }

def perform_topic_clustering(text, n_clusters=5):
    """
    Melakukan Unsupervised Learning (K-Means) untuk mengelompokkan kalimat
    berdasarkan topik secara otomatis.
    
    Args:
        text (str): Teks lengkap (Opening + Q&A).
        n_clusters (int): Jumlah cluster yang diinginkan.
        
    Returns:
        list: List of dict [{'cluster_id': int, 'label': str, 'sentences': list, 'avg_sentiment': float}]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import nltk
    from nltk.tokenize import sent_tokenize
    import numpy as np
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    # 1. Split Sentences
    sentences = sent_tokenize(text)
    # Filter short sentences
    valid_sentences = [s for s in sentences if len(s.split()) > 5]
    
    if len(valid_sentences) < n_clusters * 2:
        return [] # Not enough data
        
    # 2. Vectorization (TF-IDF)
    # Gunakan stop_words english dan max_df untuk membuang kata umum
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_sentences)
    except ValueError:
        return [] # Vocabulary empty?
        
    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    
    # 4. Extract Top Terms per Cluster & Calculate Sentiment
    feature_names = vectorizer.get_feature_names_out()
    cluster_results = []
    
    for i in range(n_clusters):
        # Get sentences in this cluster
        # Find indices where label == i
        indices = np.where(kmeans.labels_ == i)[0]
        cluster_sentences = [valid_sentences[idx] for idx in indices]
        
        # Calculate Average Sentiment
        sentiment_scores = [get_vader_score(s)['compound'] for s in cluster_sentences]
        avg_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Get Top Terms individually from centroid
        centroid = kmeans.cluster_centers_[i]
        top_indices = centroid.argsort()[-3:][::-1] # Top 3
        top_terms = [feature_names[ind] for ind in top_indices]
        
        # Label: Capitalize terms
        label = ", ".join([t.title() for t in top_terms])
        
        cluster_results.append({
            'cluster_id': i,
            'label': label,
            'count': len(cluster_sentences),
            'avg_sentiment': avg_score,
            'top_terms': top_terms
        })
        
    # Sort results by avg_sentiment for better visualization
    cluster_results.sort(key=lambda x: x['avg_sentiment'], reverse=True)
        
    return cluster_results
