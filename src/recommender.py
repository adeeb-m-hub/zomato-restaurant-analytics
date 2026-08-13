"""
recommender.py
----------------
Restaurant Recommender System.

Builds a feature matrix per restaurant (TF-IDF over cuisines + scaled
rating/votes/cost), then ranks restaurants against a user's cuisine query
using cosine similarity, combined with the restaurant's rating.

Run directly for a CLI demo:
    python src/recommender.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from data_preprocessing import load_and_preprocess


def build_feature_matrix(rec_df):
    """Build a combined TF-IDF (cuisines) + scaled-numeric feature matrix."""
    tfidf_cuisine = TfidfVectorizer(max_features=150)
    cuisine_matrix = tfidf_cuisine.fit_transform(rec_df["cuisines"].fillna(""))

    num_sparse = csr_matrix(
        rec_df[["rate_scaled", "votes_scaled", "approx_cost_scaled"]].values
    )

    feature_matrix = hstack([cuisine_matrix, num_sparse])
    return tfidf_cuisine, cuisine_matrix, feature_matrix


def recommend_restaurants(
    rec_df,
    tfidf_cuisine,
    cuisine_matrix,
    cuisine_pref,
    location_pref=None,
    budget=None,
    online_order_pref=None,
    top_n=5,
):
    """
    Two-stage restaurant recommendation.

    Stage 1 (hard filter): drop restaurants that don't match location,
    budget, or online-order preference.

    Stage 2 (soft score): vectorise the cuisine query in the same TF-IDF
    space, compute cosine similarity, and combine with rating:
        score = 0.6 * cuisine_similarity + 0.4 * (rate / 5)
    """
    filtered = rec_df.copy()

    if location_pref:
        filtered = filtered[
            filtered["location_str"].str.contains(location_pref, case=False, na=False)
        ]

    if budget:
        filtered = filtered[filtered["approx_cost"] <= budget]

    if online_order_pref:
        filtered = filtered[
            filtered["online_order_str"].str.strip().str.lower()
            == online_order_pref.strip().lower()
        ]

    if filtered.empty:
        print("No restaurants match your filters. Try relaxing some constraints.")
        return None

    query_vec = tfidf_cuisine.transform([cuisine_pref])
    filt_idx = filtered.index.tolist()
    cuisine_sub = cuisine_matrix[filt_idx]
    sims = cosine_similarity(query_vec, cuisine_sub).flatten()

    filtered = filtered.copy()
    filtered["sim"] = sims
    filtered["score"] = 0.6 * filtered["sim"] + 0.4 * (filtered["rate"] / 5.0)

    # Deduplicate by restaurant name, keep highest-scoring entry
    filtered = (
        filtered.sort_values("score", ascending=False)
        .drop_duplicates(subset=["name"], keep="first")
    )

    results = (
        filtered.head(top_n)[
            ["name", "location_str", "cuisines", "rate",
             "approx_cost", "online_order_str", "rest_type", "score"]
        ]
        .rename(columns={"location_str": "location", "online_order_str": "online_order"})
        .reset_index(drop=True)
    )
    results.index += 1
    return results


def main():
    print("Loading and preprocessing dataset (this may take a moment)...")
    rec_df = load_and_preprocess()
    tfidf_cuisine, cuisine_matrix, _ = build_feature_matrix(rec_df)

    cuisine_input = input("Enter cuisine preference (e.g. 'North Indian', 'Pizza'): ").strip()
    location_input = input("Enter area/location (or press Enter to skip): ").strip() or None
    budget_input = input("Max budget for 2 people in Rs (or press Enter to skip): ").strip()
    online_order_input = input("Want online order option? Yes / No (or press Enter to skip): ").strip() or None

    recommendations = recommend_restaurants(
        rec_df,
        tfidf_cuisine,
        cuisine_matrix,
        cuisine_pref=cuisine_input,
        location_pref=location_input,
        budget=int(budget_input) if budget_input else None,
        online_order_pref=online_order_input,
    )

    if recommendations is not None:
        print("\nTop Restaurant Recommendations:\n")
        print(recommendations)


if __name__ == "__main__":
    main()
