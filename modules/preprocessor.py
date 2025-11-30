import re

def clean_text(text):
    """
    Membersihkan teks dari metadata dan karakter yang tidak diinginkan.
    
    Args:
        text (str): Teks mentah.
        
    Returns:
        str: Teks yang sudah dibersihkan.
    """
    # Menghapus tag <NAME>...</NAME> dan isinya jika ada, atau hanya tagnya saja tergantung kebutuhan.
    # Instruksi: "menghapus metadata seperti ``, tag <NAME>...</NAME>"
    # Asumsi: Menghapus tag XML-like
    text = re.sub(r'<[^>]+>', '', text)
    
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_transcript(text):
    """
    Memisahkan transkrip menjadi Pidato Pembuka dan Sesi Tanya Jawab.
    
    Args:
        text (str): Teks transkrip lengkap.
        
    Returns:
        tuple: (opening_speech, qa_session) atau (None, None) jika separator tidak ditemukan.
    """
    # Regex Patterns yang lebih spesifik dan ketat untuk menghindari false positive
    # Masalah sebelumnya: "prepared to adjust" dianggap sebagai closing karena ada kata "prepared" ... "questions"
    # Solusi: Enforce kedekatan kata kerja (phrase-based matching)
    
    patterns = [
        # Pola 1: "I look forward to your questions"
        r"look\s+forward\s+to\s+(?:taking|answering|your)?\s*questions",
        
        # Pola 2: "happy/glad/prepared to take/answer your questions"
        # HARUS diikuti "to take" atau "to answer" agar tidak match dengan "prepared to adjust"
        r"(?:glad|happy|prepared)\s+to\s+(?:take|answer)\s+(?:your)?\s*questions",
        
        # Pola 3: "questions, please" (Fallback pendek, tapi cukup spesifik di akhir paragraf)
        r"questions\s*,?\s*please"
    ]
    
    match = None
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            break # Ambil match pertama yang valid
    
    if match:
        # Split di posisi match
        split_index = match.start()
        opening_speech = text[:split_index].strip()
        qa_session = text[split_index:].strip()
        return opening_speech, qa_session
        
    # Fallback: Cari nama moderator jika kalimat penutup tidak ketemu
    # Biasanya: <NAME>MICHELLE SMITH</NAME>
    pattern_moderator = r"<NAME>.*?MICHELLE SMITH.*?</NAME>"
    match_mod = re.search(pattern_moderator, text, re.IGNORECASE)
    
    if match_mod:
        split_index = match_mod.start()
        opening_speech = text[:split_index].strip()
        qa_session = text[split_index:].strip()
        return opening_speech, qa_session
            
    return None, None

def filter_speaker(text, target_speaker="CHAIR POWELL"):
    """
    Memfilter teks Q&A untuk hanya mengambil ucapan dari pembicara tertentu.
    Menggunakan regex untuk mendeteksi tag <NAME>...</NAME>.
    
    Args:
        text (str): Teks Q&A mentah (masih ada tag <NAME>).
        target_speaker (str): Nama pembicara yang ingin diambil (case-insensitive).
        
    Returns:
        str: Teks gabungan dari pembicara target.
    """
    # Regex penjelasan:
    # <NAME>(.*?)</NAME> : Menangkap nama pembicara di dalam tag
    # (.*?)              : Menangkap isi ucapan (non-greedy)
    # (?=<NAME>|$)       : Lookahead positif, berhenti saat ketemu tag <NAME> berikutnya atau akhir string
    pattern = r'<NAME>(.*?)</NAME>(.*?)(?=<NAME>|$)'
    
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    filtered_text = []
    
    for speaker, content in matches:
        # Bersihkan nama speaker dari spasi berlebih
        speaker = speaker.strip().upper()
        
        # Cek apakah speaker sesuai target
        if target_speaker.upper() in speaker:
            filtered_text.append(content.strip())
            
    if not filtered_text:
        # Fallback jika tidak ada match (mungkin format beda atau tidak ada tag)
        # Kembalikan teks asli jika tidak ada tag <NAME> ditemukan sama sekali
        if "<NAME>" not in text:
            return text
        return ""
        
    return " ".join(filtered_text)
