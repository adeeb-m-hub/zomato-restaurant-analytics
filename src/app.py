"""
app.py
-------
Streamlit UI for the Zomato Restaurant Assistant.

Ties together all four modules (data_preprocessing, recommender,
sentiment_analysis, chatbot) into one interactive app with a tab per
feature, so people can test each part without touching the terminal.

Run with:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd

from data_preprocessing import load_and_preprocess
from recommender import build_feature_matrix, recommend_restaurants
from sentiment_analysis import get_sia, analyze_sentiment, get_sentiment_label
from chatbot import answer_question


st.set_page_config(page_title="Zomato Restaurant Assistant", page_icon="🍽️", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders — the dataset + feature matrices only need to be built once
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def get_data():
    rec_df = load_and_preprocess()
    return rec_df


@st.cache_resource(show_spinner="Building feature matrix...")
def get_features(_rec_df):
    tfidf_cuisine, cuisine_matrix, feature_matrix = build_feature_matrix(_rec_df)
    return tfidf_cuisine, cuisine_matrix


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_model():
    return get_sia()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("🍽️ Zomato Restaurant Assistant")
st.caption(
    "A restaurant recommender, review sentiment analyzer, and Q&A chatbot — "
    "all built on the Zomato Bangalore Restaurants dataset."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app demos three ML features built on top of a cleaned "
        "Zomato Bangalore dataset:\n\n"
        "- **Recommender** — TF-IDF + cosine similarity\n"
        "- **Sentiment Analysis** — VADER\n"
        "- **Chatbot** — retrieval-based Q&A over reviews"
    )
    st.divider()
    load_clicked = st.button("Load dataset", type="primary", use_container_width=True)

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if load_clicked:
    st.session_state.data_loaded = True

if not st.session_state.data_loaded:
    st.info("Click **Load dataset** in the sidebar to get started. This runs the full cleaning pipeline once and caches it for the rest of your session.")
    st.stop()

rec_df = get_data()
tfidf_cuisine, cuisine_matrix = get_features(rec_df)
sia = get_sentiment_model()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🧹 Data Preprocessing", "🍜 Recommender", "😊 Sentiment Analysis", "💬 Chatbot"]
)

# ---------------------------------------------------------------------------
# Tab 1: Data Preprocessing
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Cleaned Dataset Preview")
    st.write(f"**{len(rec_df):,}** restaurants after cleaning (nulls dropped, phone numbers fixed, cost/rate imputed, reviews cleaned).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurants", f"{len(rec_df):,}")
    col2.metric("Avg Rating", f"{rec_df['rate'].mean():.2f}")
    col3.metric("Avg Cost (for 2)", f"₹{rec_df['approx_cost'].mean():.0f}")
    col4.metric("Unique Locations", rec_df["location_str"].nunique())

    st.dataframe(
        rec_df[["name", "location_str", "cuisines", "rate", "approx_cost", "online_order_str", "rest_type"]]
        .rename(columns={"location_str": "location", "online_order_str": "online_order"})
        .head(20),
        use_container_width=True,
    )

    st.subheader("Rating Distribution")
    st.bar_chart(rec_df["rate"].value_counts().sort_index())

    st.subheader("Cost Distribution")
    st.bar_chart(pd.cut(rec_df["approx_cost"], bins=10).value_counts().sort_index())

# ---------------------------------------------------------------------------
# Tab 2: Recommender
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Get Restaurant Recommendations")

    col1, col2, col3 = st.columns(3)
    with col1:
        cuisine_input = st.text_input("Cuisine preference", placeholder="e.g. North Indian, Pizza, Chinese")
    with col2:
        location_input = st.text_input("Location (optional)", placeholder="e.g. Koramangala")
    with col3:
        budget_input = st.number_input("Max budget for 2 (₹, optional)", min_value=0, value=0, step=50)

    online_order_input = st.radio("Online order needed?", ["No preference", "Yes", "No"], horizontal=True)

    if st.button("Get Recommendations", type="primary"):
        if not cuisine_input.strip():
            st.warning("Please enter a cuisine preference.")
        else:
            results = recommend_restaurants(
                rec_df,
                tfidf_cuisine,
                cuisine_matrix,
                cuisine_pref=cuisine_input,
                location_pref=location_input or None,
                budget=budget_input if budget_input > 0 else None,
                online_order_pref=None if online_order_input == "No preference" else online_order_input,
            )
            if results is None:
                st.error("No restaurants match your filters. Try relaxing some constraints.")
            else:
                st.session_state["last_recommendations"] = results["name"].tolist()
                st.success(f"Found {len(results)} matches:")
                st.dataframe(results, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Sentiment Analysis
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Restaurant Review Sentiment")
    st.write("Analyze sentiment for restaurants from your last recommendation search, or type a name directly.")

    default_names = st.session_state.get("last_recommendations", [])
    restaurant_names_input = st.text_area(
        "Restaurant name(s) — one per line",
        value="\n".join(default_names[:5]) if default_names else "",
        placeholder="e.g.\nTruffles\nOnesta",
    )

    if st.button("Analyze Sentiment", type="primary"):
        names = [n.strip() for n in restaurant_names_input.splitlines() if n.strip()]
        if not names:
            st.warning("Enter at least one restaurant name, or run a recommendation search first.")
        else:
            summary = analyze_sentiment(rec_df, names, sia=sia)
            if summary.empty:
                st.error("No matching restaurants found for those names.")
            else:
                st.dataframe(summary, use_container_width=True)

                for _, row in summary.iterrows():
                    label = row["overall_sentiment"]
                    emoji = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}.get(label, "⚪")
                    st.write(f"{emoji} **{row['name']}** — {label} (score: {row['avg_score']:.2f})")

# ---------------------------------------------------------------------------
# Tab 4: Chatbot
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Ask About a Restaurant")
    st.write("Retrieval-based chatbot — matches your question against real reviews for the restaurant you pick.")

    default_names = st.session_state.get("last_recommendations", [])
    if default_names:
        restaurant_choice = st.selectbox("Restaurant", default_names)
    else:
        restaurant_choice = st.text_input("Restaurant name", placeholder="e.g. Truffles")

    question = st.text_input("Your question", placeholder="e.g. Is it good for families? How's the service?")

    if st.button("Ask", type="primary"):
        if not restaurant_choice or not question.strip():
            st.warning("Enter both a restaurant name and a question.")
        else:
            rows = rec_df[rec_df["name"].str.lower() == restaurant_choice.lower()]
            if rows.empty:
                st.error(f"No restaurant named '{restaurant_choice}' found. Check the spelling.")
            else:
                import re
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                unique_reviews = rows["reviews_list"].dropna().astype(str).unique().tolist()
                all_reviews = " ".join(unique_reviews)
                snippets = re.split(r"[./\n]+", all_reviews)
                snippets = [s.strip() for s in snippets if len(s.strip()) > 20]

                seen, unique_snippets = set(), []
                for s in snippets:
                    if s.lower() not in seen:
                        seen.add(s.lower())
                        unique_snippets.append(s)
                snippets = unique_snippets

                if not snippets:
                    st.warning("Not enough review text to answer this question.")
                else:
                    corpus = snippets + [question]
                    vectorizer = TfidfVectorizer(stop_words="english")
                    tfidf_mat = vectorizer.fit_transform(corpus)
                    sims = cosine_similarity(tfidf_mat[-1], tfidf_mat[:-1]).flatten()
                    top_indices = sims.argsort()[::-1][:6]

                    st.write(f"**Question:** {question}")
                    shown = 0
                    seen_out = set()
                    for idx in top_indices:
                        snippet = snippets[idx].strip().capitalize()
                        if snippet in seen_out or sims[idx] < 0.01:
                            continue
                        st.write(f"- {snippet}")
                        seen_out.add(snippet)
                        shown += 1
                        if shown >= 3:
                            break
                    if shown == 0:
                        st.info("Couldn't find relevant info — try asking about food, service, or ambience.")
