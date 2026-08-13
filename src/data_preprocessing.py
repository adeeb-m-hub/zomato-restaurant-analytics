"""
data_preprocessing.py
----------------------
Loads the Zomato Bangalore Restaurants dataset and runs the full cleaning /
preprocessing pipeline: handling missing values, fixing inconsistent formats,
imputation, encoding, and feature scaling.

This module is shared by recommender.py, sentiment_analysis.py, and
chatbot.py — each of them calls `load_and_preprocess()` to get a ready-to-use
DataFrame instead of repeating the cleaning steps.
"""

import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def clean_reviews(x):
    """Clean a single raw reviews_list entry: strip brackets, ratings text,
    symbols, and short/meaningless fragments."""
    if pd.isnull(x):
        return []

    x = str(x)
    x = x.replace("\\n", " ").replace("\\r", " ")
    x = re.sub(r"Rated\s*\d+\.?\d*", "", x)
    x = re.sub(r"[\[\]\(\)\'\"]", "", x)

    parts = x.split(",")
    cleaned = []
    for part in parts:
        part = re.sub(r"\brated\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"[^\w\s]", "", part)
        part = re.sub(r"\s+", " ", part)
        part = part.lower().strip()
        if len(part) > 3:
            cleaned.append(part)

    return cleaned


def extract_two_numbers(x):
    """Pull one or two valid phone numbers out of a messy phone string."""
    if pd.isnull(x) and str(x).strip() == "":
        return np.nan

    x = str(x)
    numbers = re.findall(r"\d+", x)
    valid_numbers = [num for num in numbers if len(num) in [10, 11]]

    if len(valid_numbers) >= 2:
        return f"{valid_numbers[0]}/{valid_numbers[1]}"
    elif len(valid_numbers) == 1:
        return valid_numbers[0]
    else:
        return np.nan


def strip_mojibake(text):
    """Remove garbled UTF-8/Latin-1 mojibake and stray hex remnants left
    behind in the reviews text after scraping."""
    if pd.isnull(text) or not str(text).strip():
        return ""

    text = str(text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"x[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_dataset():
    """Load the raw Zomato Bangalore Restaurants dataset via kagglehub.

    Requires a Kaggle account + API token configured locally
    (see README for setup instructions).
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    file_path = "zomato.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "himanshupoddar/zomato-bangalore-restaurants",
        file_path,
    )
    return df


def load_and_preprocess():
    """Run the full cleaning + preprocessing pipeline and return a DataFrame
    ready for the recommender, sentiment analysis, and chatbot modules.

    Returns
    -------
    rec_df : pd.DataFrame
        Cleaned, encoded, and scaled dataset, with `location` and
        `online_order` also kept in human-readable string form
        (`location_str`, `online_order_str`) for filtering.
    """
    df = load_raw_dataset()
    original_df = df.copy()

    # --- Basic string columns ---
    str_cols = ["url", "address", "name"]
    df[str_cols] = df[str_cols].astype("string")

    # --- rate: "4.1/5" -> 4.1, "-"/"NEW" -> NaN ---
    df["rate"] = df["rate"].replace(["-", "NEW"], np.nan)
    df["rate"] = df["rate"].apply(
        lambda x: float(str(x).split("/")[0]) if pd.notnull(x) else np.nan
    )

    # --- phone: strip formatting, extract valid numbers, drop unrecoverable ---
    df["phone"] = df["phone"].str.replace(r"^080\s+", "080", regex=True)
    df["phone"] = df["phone"].str.replace(r"^\+91", "", regex=True)
    df["phone"] = df["phone"].str.strip()
    df["phone"] = df["phone"].apply(extract_two_numbers)
    df = df.dropna(subset=["phone"]).reset_index(drop=True)

    # --- rest_type: strip whitespace, drop nulls ---
    df["rest_type"] = df["rest_type"].apply(
        lambda x: str(x).strip() if pd.notnull(x) else np.nan
    )
    df = df.dropna(subset=["rest_type"]).reset_index(drop=True)

    # --- dish_liked: fill nulls, ~half the column is missing ---
    df["dish_liked"] = df["dish_liked"].apply(
        lambda x: str(x).strip() if pd.notnull(x) else np.nan
    )
    df["dish_liked"] = df["dish_liked"].fillna("Dish not liked")

    # --- cuisines: strip whitespace, drop nulls ---
    df["cuisines"] = df["cuisines"].apply(
        lambda x: str(x).strip() if pd.notnull(x) else np.nan
    )
    df = df.dropna(subset=["cuisines"]).reset_index(drop=True)

    # --- approx_cost: remove commas, convert to numeric ---
    df = df.rename(columns={"approx_cost(for two people)": "approx_cost"})
    df["approx_cost"] = df["approx_cost"].replace(",", "", regex=True)
    df["approx_cost"] = pd.to_numeric(df["approx_cost"])

    # --- reviews_list: clean text, then strip mojibake ---
    df["reviews_list"] = df["reviews_list"].apply(clean_reviews)
    df["reviews_list"] = df["reviews_list"].apply(
        lambda x: "/".join([review.capitalize() for review in x])
    )

    # --- drop menu_item (mostly empty, not useful) ---
    df = df.drop(columns=["menu_item"])

    # --- final dtypes ---
    cols = ["phone", "rest_type", "dish_liked", "cuisines", "reviews_list"]
    df[cols] = df[cols].astype("string")

    cat_cols = ["online_order", "book_table", "location", "listed_in(type)", "listed_in(city)"]
    df[cat_cols] = df[cat_cols].astype("category")

    # --- impute rate & approx_cost with median (skewed distributions) ---
    num_cols = ["rate", "approx_cost"]
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    # --- encode categoricals ---
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # --- scale numeric features ---
    df[["rate_scaled"]] = StandardScaler().fit_transform(df[["rate"]])
    df[["votes_scaled"]] = MinMaxScaler().fit_transform(df[["votes"]])
    df[["approx_cost_scaled"]] = MinMaxScaler().fit_transform(df[["approx_cost"]])

    # --- pull location/online_order back as human-readable strings ---
    orig_str = original_df[["name", "address", "location", "online_order"]].copy()
    orig_str["location_str"] = orig_str["location"].astype(str).str.strip()
    orig_str["online_order_str"] = orig_str["online_order"].astype(str).str.strip()
    orig_str = orig_str[["name", "address", "location_str", "online_order_str"]]
    orig_str = orig_str.drop_duplicates(subset=["name", "address"])

    rec_df = df.merge(orig_str, on=["name", "address"], how="left")

    # --- strip mojibake from reviews (emojis / garbled scraped characters) ---
    rec_df["reviews_list"] = rec_df["reviews_list"].apply(strip_mojibake)

    return rec_df
