import random
from transformers import pipeline
from scipy import stats
import plotly.graph_objects as go
import numpy as np

class ScientificValidator:
    """
    Validasi ilmiah membandingkan skor VADER modifikasi dengan model SOTA (FinBERT).
    """
    
    def __init__(self):
        """
        Inisialisasi model FinBERT.
        Menggunakan pipeline 'sentiment-analysis' dari Hugging Face.
        """
        print("Loading FinBERT model for validation (this may take a while)...")
        # Load SOTA Model: ProsusAI/finbert (Specific for Financial Sentiment)
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
        
    def validate_against_sota(self, texts, vader_scores, sample_size=30):
        """
        Membandingkan skor VADER dengan FinBERT pada sampel acak.
        
        Args:
            texts (list): List kalimat original.
            vader_scores (list): List skor VADER (compound) yang sudah dihitung.
            sample_size (int): Jumlah sampel untuk validasi (default 30 agar cepat).
            
        Returns:
            dict: Hasil statistik korelasi dan data plotting.
        """
        # Ensure consistent length
        if len(texts) != len(vader_scores):
            raise ValueError("Length of texts and scores must match")
            
        # Select random indices
        total_len = len(texts)
        if total_len == 0:
            return {'correlation': 0, 'p_value': 1, 'data': []}
            
        indices = random.sample(range(total_len), min(sample_size, total_len))
        
        validation_data = []
        finbert_scores = []
        sampled_vader = []
        
        for idx in indices:
            text = texts[idx]
            v_score = vader_scores[idx]
            
            # Run FinBERT
            # Output format: [[{'label': 'positive', 'score': 0.9}, ...]]
            # FinBERT labels: 'positive', 'negative', 'neutral'
            try:
                results = self.finbert(text)[0]
                
                # Extract probabilities
                prob_pos = next((item['score'] for item in results if item['label'] == 'positive'), 0)
                prob_neg = next((item['score'] for item in results if item['label'] == 'negative'), 0)
                # prob_neu = next((item['score'] for item in results if item['label'] == 'neutral'), 0)
                
                # Convert to single scalar score (-1 to 1)
                # Logic: Positive Prob - Negative Prob (Neutral ignored as 0 center)
                f_score = prob_pos - prob_neg
                
                finbert_scores.append(f_score)
                sampled_vader.append(v_score)
                
                validation_data.append({
                    'text': text,
                    'vader_score': v_score,
                    'finbert_score': f_score
                })
            except Exception as e:
                print(f"Error processing text for FinBERT: {e}")
                
        # Calculate Pearson Correlation
        if len(validation_data) > 1:
            corr_coef, p_value = stats.pearsonr(sampled_vader, finbert_scores)
        else:
            corr_coef, p_value = 0.0, 1.0
            
        return {
            'correlation': corr_coef,
            'p_value': p_value,
            'data': validation_data
        }
        
    def plot_validation_scatter(self, validation_result):
        """
        Membuat Scatter Plot perbandingan VADER vs FinBERT.
        """
        data = validation_result['data']
        corr = validation_result['correlation']
        
        vader_vals = [d['vader_score'] for d in data]
        finbert_vals = [d['finbert_score'] for d in data]
        texts = [d['text'][:100] + "..." for d in data] # Truncate for hover
        
        fig = go.Figure()
        
        # Scatter Points
        fig.add_trace(go.Scatter(
            x=vader_vals,
            y=finbert_vals,
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.6),
            text=texts,
            name='Sampled Sentences'
        ))
        
        # Diagonal Line (Ideal Connection)
        fig.add_shape(
            type="line",
            x0=-1, y0=-1,
            x1=1, y1=1,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.update_layout(
            title=f"Validasi Ilmiah: Modifikasi VADER vs FinBERT (SOTA)<br><sub>Pearson Correlation: {corr:.4f}</sub>",
            xaxis_title="Skor VADER Modifikasi",
            yaxis_title="Skor FinBERT (Pos - Neg Prob)",
            template="plotly_white",
            width=700,
            height=600,
            xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1])
        )
        
        return fig
