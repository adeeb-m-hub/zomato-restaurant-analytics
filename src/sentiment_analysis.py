"""
sentiment_analysis.py
-----------------------
Customer Sentiment Analysis using VADER (Valence Aware Dictionary and
sEntiment Reasoner) on restaurant reviews.

Compound score  ->  Label
   >= 0.05       ->  Positive
  <= -0.05       ->  Negative
   otherwise     ->  Neutral

Run directly for a CLI demo (recommends restaurants first, then scores
their reviews):
    python src/sentiment_analysis.py
"""

import pandas as pd
import nltk

from data_preprocessing import load_and_preprocess
from recommender import build_feature_matrix, recommend_restaurants


def get_sia():
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def get_sentiment_label(text, sia):
    if not text or pd.isnull(text):
        return "No Review"
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_sentiment_score(text, sia):
    if not text or pd.isnull(text):
        return 0.0
    return sia.polarity_scores(str(text))["compound"]


def analyze_sentiment(rec_df, restaurant_names, sia=None):
    """Return a per-restaurant sentiment summary (dominant label + avg score)
    for the given list of restaurant names."""
    if sia is None:
        sia = get_sia()

    rows = rec_df[rec_df["name"].isin(restaurant_names)].copy()
    rows["sentiment"] = rows["reviews_list"].apply(lambda t: get_sentiment_label(t, sia))
    rows["sentiment_score"] = rows["reviews_list"].apply(lambda t: get_sentiment_score(t, sia))

    summary = (
        rows.groupby("name", as_index=False)
        .agg(
            overall_sentiment=("sentiment", lambda x: x.mode()[0]),
            avg_score=("sentiment_score", "mean"),
        )
    )
    return summary


def main():
    print("Loading and preprocessing dataset (this may take a moment)...")
    rec_df = load_and_preprocess()
    tfidf_cuisine, cuisine_matrix, _ = build_feature_matrix(rec_df)

    cuisine_input = input("Enter cuisine preference (e.g. 'North Indian', 'Pizza'): ").strip()
    recommendations = recommend_restaurants(
        rec_df, tfidf_cuisine, cuisine_matrix, cuisine_pref=cuisine_input
    )

    if recommendations is None:
        return

    print("\nRunning sentiment analysis on recommended restaurants...\n")
    summary = analyze_sentiment(rec_df, recommendations["name"].tolist())
    print(summary)


if __name__ == "__main__":
    main()
