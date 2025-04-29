from abc import ABC, abstractmethod

import pandas as pd
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


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

    @abstractmethod
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
        raise NotImplementedError


class MySentimentIntensityAnalyzer(SentimentBase):
    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()

    def classify(self, text: str) -> float:
        text = self.preprocessing(text)
        scores = self.classifier.polarity_scores(text)
        assert -1 <= scores['compound'] <= 1
        return scores['compound']

    def preprocessing(self, text: str) -> str:
        text = text.replace("#", "")
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


def visualize_results(results: list) -> None:
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
        title="Sentiment Analysis Scatter Plot"
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

if __name__ == "__main__":
    classifier = MySentimentIntensityAnalyzer()
    results = [analyze_sentiment(text, classifier) for text in sample_data]
    visualize_results(results)
