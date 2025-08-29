import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(reviews_expanded):
    df = reviews_expanded.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["rank"] = df.groupby("parent_asin")["rolling_reviews"].rank(method="first", ascending=False)
    df = df[df["rank"] <= 3].copy()

    # Seasonal features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    X = df[["tot_desc", "sin_day", "cos_day"]]
    y = df["rolling_reviews"].values

    return train_test_split(X, y, test_size=0.2, random_state=42)
