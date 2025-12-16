# Analysis of Sentiment Dynamics and Market Response in FOMC Press Conferences Using Enhanced VADER Method

A Python-based sentiment analysis tool designed to analyze press conference transcripts from the **Federal Open Market Committee (FOMC)**. This project aims to assist investors and researchers in understanding the *tonality* (Hawkish vs. Dovish) of Federal Reserve Chair Jerome Powell's speeches and correlating them with market movements (S&P 500).

## Key Features

*   **Calibrated VADER Sentiment Analysis**: Utilizes the VADER algorithm customized with a *Financial Lexicon* for high accuracy in economic contexts.
*   **Domain Adaptation**: Implements specific weight adjustments for directional words (e.g., "lower", "cut" are neutralized) to avoid false negatives, ensuring that sentiment reflects the economic context rather than individual word scores.
*   **Smart Context Logic (spaCy)**: Employs advanced NLP techniques (*Dependency Parsing*) to interpret economic phrases correctly (e.g., *"Inflation falls"* is interpreted as Positive, while *"Growth slows"* is Negative).
*   **Topic Clustering**: Automatically identifies and groups key topics using **K-Means Clustering**, allowing for granular sentiment analysis per topic (e.g., Inflation, Labor Market).
*   **Market Response Correlation**: Integrates with **S&P 500** data (via `yfinance`) to analyze the correlation between Powell's sentiment and immediate market reactions.
*   **Speaker Filtering**: Automatically filters Q&A sessions to isolate responses from Chair Powell, removing noise from reporters and moderators using robust Regex patterns.
*   **Multi-Audience Interpretation**: Provides analysis results tailored for two distinct audiences:
    *   **Investors**: Focuses on market terminology (Risk-On/Off, Hawkish/Dovish).
    *   **General Public**: Explains economic impacts in everyday language.
*   **Interactive Visualization**:
    *   **Sentiment Flow**: Line chart visualizing the sentence-by-sentence sentiment progression.
    *   **Historical Trend**: Tracks sentiment trends from 2020-2025 with markers for the currently analyzed file and market correlation scatter plots.
    *   **Word Cloud**: Visualizes dominant keywords with POS Tagging filters.

## Methodology

This project combines a **Lexicon-Based** approach (VADER) with **Rule-Based NLP** (spaCy) and **Statistical Analysis**.

1.  **Preprocessing**:
    *   **Splitting**: Robust separation of Opening Speech and Q&A Session using flexible Regex patterns.
    *   **Filtering**: Isolation of Chair Powell's speech in the Q&A session.
2.  **Smart Context**: Transformation of economic phrases into single sentiment tokens (e.g., `lower inflation` -> `economic_positive`).
3.  **Scoring**: Calculation of the VADER *Compound* score.
4.  **Clustering**: Unsupervised grouping of sentences into topics using TF-IDF and K-Means.
5.  **Market Correlation**: Pearson correlation analysis between sentiment scores and S&P 500 percentage changes.
6.  **Interpretation**: Conversion of numerical scores into narrative insights.

## Installation

1.  **Clone Repository**
    ```bash
    git clone https://github.com/iqbalmo/fomc-vader.git
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *The program will automatically download the spaCy model (`en_core_web_sm`) and NLTK data upon first run.*

## Usage

Run the application using Streamlit:

```bash
python -m streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

## Project Structure

```
fomc-vader/
├── app.py                  # Main Entry Point (Streamlit UI)
├── requirements.txt        # Python Dependencies
├── modules/                # Logic Modules
│   ├── preprocessor.py     # Text Cleaning, Splitting & Filtering
│   ├── analyzer.py         # Sentiment Logic, Clustering & Market Analysis
│   └── visualizer.py       # Visualization (Plotly + WordCloud)
└── fomc-transcript/        # Transcript Dataset (.txt)
```

---
*Developed for Thesis: Federal Reserve Tonality Sentiment Analysis.*
