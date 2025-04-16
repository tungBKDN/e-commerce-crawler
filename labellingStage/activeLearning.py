import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import utils as u

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
from typing import List, Tuple


class EmotionClassifierNB:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(),
            MultinomialNB()
        )
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classifier on preprocessed texts and their labels.
        """
        print(f"Length of texts: {len(texts)}")
        print(f"Length of labels: {len(labels)}")
        self.model.fit(texts, labels)
        self.is_trained = True
        print("✅ Model trained on", len(texts), "samples.")

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict emotion label and confidence for a list of texts.
        Returns a list of (label, confidence) tuples.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call `fit()` first.")

        probas = self.model.predict_proba(texts)
        predicted_labels = self.model.predict(texts)
        confidence_scores = probas.max(axis=1)

        return list(zip(predicted_labels, confidence_scores))

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "comment") -> pd.DataFrame:
        """
        Apply prediction to a DataFrame and add 'predicted_label' and 'confidence' columns.
        """
        if text_col not in df.columns:
            raise ValueError(f"'{text_col}' column not found in the DataFrame.")

        idx = df[df['labeled'] != True].index
        texts = df.loc[idx, text_col].tolist()
        predictions = self.predict(texts)
        predicted_labels, confidence_scores = zip(*predictions)
        df.loc[idx, 'predicted_label'] = predicted_labels
        df.loc[idx, 'confidence'] = confidence_scores
        return df
