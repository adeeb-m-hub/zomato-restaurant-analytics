# Zomato Restaurant Assistant 🍽️

A data science pipeline and retrieval assistant built on the [Zomato Bangalore Restaurants dataset](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants). Covers the full journey from messy raw data to three working ML-powered features.

## Features

- **🧹 Data Preprocessing Pipeline** — cleaning, handling missing values, imputation, encoding, and feature scaling on a genuinely messy real-world dataset
- **📊 EDA** — univariate/bivariate analysis, distribution plots, correlation heatmaps
- **🍜 Restaurant Recommender** — suggests restaurants based on cuisine, location, and budget using TF-IDF + cosine similarity
- **😊 Sentiment Analysis** — classifies restaurant reviews as Positive / Neutral / Negative using VADER
- **💬 Review-Based Q&A Chatbot** — retrieval-based chatbot that answers questions about a restaurant by matching them against its reviews (no LLM required)

## Project Structure

```
zomato-restaurant-assistant/
├── src/
│   ├── data_preprocessing.py   # Shared cleaning + preprocessing pipeline
│   ├── recommender.py          # Restaurant recommender (TF-IDF + cosine similarity)
│   ├── sentiment_analysis.py   # VADER-based review sentiment analysis
│   └── chatbot.py               # Retrieval-based Q&A chatbot
├── notebooks/
│   └── full_pipeline.ipynb     # Original end-to-end notebook (EDA + all 3 features)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd zomato-restaurant-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   That's it — the dataset is public, so kagglehub downloads it automatically the first time you run any module, no Kaggle account or API key needed.

## Usage

Each module can be run standalone as a CLI demo:

```bash
cd src

# Get restaurant recommendations
python recommender.py

# Get recommendations + sentiment breakdown of their reviews
python sentiment_analysis.py

# Get recommendations, then ask a chatbot questions about one of them
python chatbot.py
```

Or import the pieces directly in your own script:

```python
from data_preprocessing import load_and_preprocess
from recommender import build_feature_matrix, recommend_restaurants

rec_df = load_and_preprocess()
tfidf_cuisine, cuisine_matrix, _ = build_feature_matrix(rec_df)

results = recommend_restaurants(
    rec_df, tfidf_cuisine, cuisine_matrix,
    cuisine_pref="North Indian", location_pref="Koramangala", budget=800
)
print(results)
```

## How It Works

**Recommender:** Each restaurant is represented as a feature vector combining a TF-IDF encoding of its cuisines with scaled rating/votes/cost. User filters (location, budget, online order) are applied first as a hard filter, then remaining restaurants are ranked by `0.6 × cuisine_similarity + 0.4 × (rating / 5)`.

**Sentiment Analysis:** Runs VADER's compound polarity score over each restaurant's cleaned reviews and aggregates them into an overall label and average score.

**Chatbot:** Splits a restaurant's reviews into sentence-level snippets, vectorizes them alongside the user's question with TF-IDF, and returns the snippets with the highest cosine similarity — a lightweight retrieval approach that needs no LLM.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `nltk` (VADER) · `matplotlib` / `seaborn` · `kagglehub`
