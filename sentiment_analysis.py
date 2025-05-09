import argparse
import re
from abc import ABC, abstractmethod

import pandas as pd
import plotly.express as px
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)


class SentimentBase(ABC):

    @abstractmethod
    def classify(self, text: str) -> float:
        """
        Analyzes the sentiment of the given text and return its score.

        Args:
            text (str): The input text to analyze.

        Returns:
            float: A confidence scores with values in the range [-1, 1].

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def preprocessing(self, text: str) -> str:
        """
        Preprocesses the input text by applying necessary transformations.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.

        Raises:
            NotImplementedError
        """
        # Remove links from the text
        text = re.sub(r'http\S+', '', text)
        # Remove extra spaces from the text
        text = re.sub(r'\s+', ' ', text)
        return text


class MySentimentIntensityAnalyzer(SentimentBase):
    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()

    def classify(self, text: str) -> float:
        text = self.preprocessing(text)
        scores = self.classifier.polarity_scores(text)
        assert -1 <= scores['compound'] <= 1
        return scores['compound']

    def preprocessing(self, text: str) -> str:
        text = super().preprocessing(text)
        text = text.replace("#", "")
        return text


class MyFlairSentimentClassifier(SentimentBase):
    def __init__(self):
        self.classifier = TextClassifier.load('en-sentiment')

    def classify(self, text: str) -> float:
        text = self.preprocessing(text)
        sentence = Sentence(text)
        self.classifier.predict(sentence)
        score = sentence.score if sentence.tag == 'POSITIVE' else -sentence.score
        assert -1 <= score <= 1
        return score

    def preprocessing(self, text: str) -> str:
        text = super().preprocessing(text)
        return text


class MySiEBERT(SentimentBase):
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis",
                                   model="siebert/sentiment-roberta-large-english")

    def classify(self, text: str) -> float:
        text = self.preprocessing(text)
        pred = self.classifier(text)
        score = pred[0]['score']\
            if pred[0]['label'] == 'POSITIVE'\
            else -pred[0]['score']
        assert -1 <= score <= 1
        return score

    def preprocessing(self, text: str) -> str:
        super().preprocessing(text)
        return text


class MyroBERTa(SentimentBase):
    def __init__(self):
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def classify(self, text: str) -> float:
        text = self.preprocessing(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score = (scores[2] - scores[0]) * (1 - scores[1])
        assert -1 <= score <= 1
        return score

    def preprocessing(self, text: str) -> str:
        super().preprocessing(text)
        return text


def analyze_sentiment(text: str,
                      classifier: SentimentBase,
                      threshold: float = 0.05) -> dict:
    """
    Analyzes the sentiment of the given text and classifies it as positive, negative, or neutral.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the sentiment classification and confidence scores.
    """
    score = classifier.classify(text)

    if score > threshold:
        sentiment = 'Positive'
    elif score < -threshold:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return {
        'text': text,
        'sentiment': sentiment,
        'score': score
    }


def visualize_results(results: list, title: str | None = None) -> None:
    """
    Visualizes the sentiment analysis results using a bar chart.

    Args:
        results (list): A list of dictionaries containing sentiment analysis results.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = pd.DataFrame(results)
    embeddings = model.encode(results['text'], normalize_embeddings=True)
    embeddings = PCA(n_components=2).fit_transform(embeddings)
    fig = px.scatter(
        results,
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        color="sentiment",
        hover_data=["text", "score"],
        title=title or "Sentiment Analysis Scatter Plot"
    )
    fig.show()


sample_data = [
    "The new interface is amazing and intuitive!",  # Positive
    "This product is terrible, nothing works properly.",  # Negative
    "Received the package yesterday, it's okay but not what I expected.",  # Neutral
    "I've been using this for a week and haven't had any issues.",  # Neutral
    "Customer service never responded to my emails.",  # Negative
    "Average performance for the price point.",  # Neutral
    "Absolutely love this product, would recommend to everyone!",  # Positive
    "The mobile app keeps crashing on my phone.",  # Negative
    "It's fine, does what it's supposed to do.",  # Neutral
    "Completely disappointed with the quality."  # Negative
]


def parser():
    parser = argparse.ArgumentParser(description='Text Sentiment Analysis')
    parser.add_argument('--model', type=str, help='Select model to use for inference',
                        choices=['flair', 'nltk', 'siebert', 'roberta'], default='roberta')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Threshold value for sentiment classification')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all models and visualize results')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    classifiers = {'flair': MyFlairSentimentClassifier(),
                   'nltk': MySentimentIntensityAnalyzer(),
                   'siebert': MySiEBERT(),
                   'roberta': MyroBERTa()}
    for name, classifier in classifiers.items():
        if name != args.model and not args.run_all:
            continue
        results = [analyze_sentiment(text, classifier, args.threshold)
                   for text in sample_data]
        visualize_results(results, title=f'Sentiment Analysis with {name}')
