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
    separators = [
        "I look forward to your questions",
        "glad to take your questions",
        "questions, please"
    ]
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)
            # Bagian pertama adalah opening, bagian kedua adalah Q&A (termasuk separatornya atau tidak? 
            # Biasanya separator adalah penutup opening atau pembuka Q&A. 
            # Kita masukkan separator ke bagian opening atau membuangnya. 
            # Untuk kebersihan, kita bisa membuangnya atau membiarkannya.
            # Mari kita pisahkan tepat di separator.
            opening_speech = parts[0].strip()
            qa_session = sep + parts[1] # Memasukkan separator ke Q&A agar konteks tetap ada, atau bisa dibuang.
            # Instruksi: "opening_speech (sebelum frasa) dan qa_session (setelah frasa)"
            # Mari ikuti instruksi ketat: sebelum dan setelah.
            qa_session = parts[1].strip()
            
            return opening_speech, qa_session
            
    return None, None
