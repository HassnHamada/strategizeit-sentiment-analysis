# Sentiment Analysis Tool

## Overview

This project implements a simple sentiment analysis tool using different methods:
- NLTK's [`SentimentIntensityAnalyzer`](https://www.nltk.org/api/nltk.sentiment.SentimentIntensityAnalyzer.html)
- Flair's `TextClassifier`
- [`SiEBERT`](https://huggingface.co/siebert/sentiment-roberta-large-english)
- [`roBERTa`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)

The tool processes customer feedback and classifies the sentiment of each text as **POSITIVE**, **NEGATIVE**, or **NEUTRAL**. Additionally, it provides confidence scores for each classification and visualises them in an interactive window.
## Approach

1. **Classification Logic**:
   - A score is calculated for each text between -1 (most negative) and 1 (most positive).
   - Thresholds are applied to classify the sentiment.
3. **Visualisation**:
   - Convert the text to vector embeddings using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
   - Use PCA to reduce the dimensionality of the embeddings to 2 dimensions.
   - Use Plotly to create an interactive scatter plot.

## Preprocessing Steps

TODO

## Limitations

1. **Mixed Sentiments**: Texts with mixed sentiments may not be classified accurately.
2. **Short Texts**: Very short texts may lack sufficient information for accurate classification.

## Potential Improvements

1. **Custom Training**: Train a custom sentiment analysis model on domain-specific data.
2. **Advanced Models**: Use modern transformer-based models like GPT for better context understanding.
3. **Services**: Use services like [AWS Comprehend](https://aws.amazon.com/comprehend/features/) for sentiment analysis.
4. **Embeddings**: Use state-of-the-art vector embeddings like [openai](https://platform.openai.com/docs/models/text-embedding-3-large) for better visualisation.

## Instructions for Running the Tool

1. Ensure you have Python 3.13.2 installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/HassnHamada/strategizeit-sentiment-analysis.git
   ```
3. Navigate to the project directory:
   ```bash
   cd strategizeit-sentiment-analysis
   ```
4. Start a new virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
6. Run the script:
   ```bash
   python sentiment_analysis.py
   ```
   The default inference model is `roberta`.

   Other available options:
      - `nltk` to use NLTK's SentimentIntensityAnalyzer
      - `flair` to use Flair's TextClassifier
      - `siebert` to use SiEBERT
      - `roberta` to use roBERTa (default)

   example, to use Flair:
   ```bash
   python sentiment_analysis.py --model flair
   ```
7. Enjoy

## Note

- If you are on MacOS, you may need to install the `sentencepiece` package.

  ```bash
  brew install sentencepiece
  ```
- Problems with `flair` and `siebert`:
  - It does not have a **NEUTRAL** class.
  - The confidence is often skewed due to fine-tuning on polarized data.
