"""
chatbot.py
-----------
Mini Review-Based Q&A Chatbot.

Retrieval-based approach (no LLM needed):
1. Fetches all reviews for a chosen restaurant.
2. Splits them into sentence-level snippets.
3. Vectorises snippets + the user's question with TF-IDF.
4. Returns the top snippets ranked by cosine similarity to the question.

Run directly for a CLI demo:
    python src/chatbot.py
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import load_and_preprocess
from recommender import build_feature_matrix, recommend_restaurants


def answer_question(rec_df, restaurant_name, question, top_k=3):
    """Answer a natural-language question about a restaurant by retrieving
    the most relevant review snippets (TF-IDF + cosine similarity)."""
    rows = rec_df[rec_df["name"].str.lower() == restaurant_name.lower()]

    if rows.empty:
        print(f"No restaurant named '{restaurant_name}' found. Check the spelling.")
        return

    # Deduplicate reviews before splitting to avoid inflating repeated snippets
    unique_reviews = rows["reviews_list"].dropna().astype(str).unique().tolist()
    all_reviews = " ".join(unique_reviews)

    snippets = re.split(r"[./\n]+", all_reviews)
    snippets = [s.strip() for s in snippets if len(s.strip()) > 20]

    seen_snippets = set()
    unique_snippets = []
    for s in snippets:
        if s.lower() not in seen_snippets:
            seen_snippets.add(s.lower())
            unique_snippets.append(s)
    snippets = unique_snippets

    if not snippets:
        print("Not enough review text to answer this question.")
        return

    corpus = snippets + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_mat = vectorizer.fit_transform(corpus)
        query_vec = tfidf_mat[-1]
        snip_vecs = tfidf_mat[:-1]
        sims = cosine_similarity(query_vec, snip_vecs).flatten()
    except Exception as e:
        print(f"Could not process reviews: {e}")
        return

    top_indices = sims.argsort()[::-1][: top_k * 2]

    print(f"\nChatbot Answer for '{restaurant_name}'")
    print(f"    Question: {question}\n")

    shown, seen = 0, set()
    for idx in top_indices:
        snippet = snippets[idx].strip().capitalize()
        if snippet in seen or len(snippet) < 20 or sims[idx] < 0.01:
            continue
        print(f"  {snippet}")
        seen.add(snippet)
        shown += 1
        if shown >= top_k:
            break

    if shown == 0:
        print("  Couldn't find relevant info - try asking about food, service, or ambience.")


def main():
    print("Loading and preprocessing dataset (this may take a moment)...")
    rec_df = load_and_preprocess()
    tfidf_cuisine, cuisine_matrix, _ = build_feature_matrix(rec_df)

    cuisine_input = input("Enter cuisine preference (e.g. 'North Indian', 'Pizza'): ").strip()
    recommendations = recommend_restaurants(
        rec_df, tfidf_cuisine, cuisine_matrix, cuisine_pref=cuisine_input
    )

    if recommendations is not None:
        print("\nRestaurants you can ask about:")
        for name in recommendations["name"].tolist():
            print(f"  * {name}")

    restaurant_choice = input("\nEnter restaurant name to ask about: ").strip()
    print("\nAsk anything about this restaurant. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ["exit", "quit", "no", "stop"]:
            break
        if not user_question:
            print("Please enter a question.")
            continue
        answer_question(rec_df, restaurant_choice, user_question)
        print()

    print("\nThanks for using the Restaurant Assistant!")


if __name__ == "__main__":
    main()
