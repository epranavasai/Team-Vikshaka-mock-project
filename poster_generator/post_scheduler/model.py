import joblib # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor # type: ignore

def build_pipeline():
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)),
        ("svd", TruncatedSVD(n_components=128, random_state=42))
    ])

    preprocessor = ColumnTransformer([
        ("text", text_pipeline, "tot_desc"),
        ("cyclic", "passthrough", ["sin_day", "cos_day"])
    ])

    model = Pipeline([
        ("features", preprocessor),
        ("xgb", XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            tree_method="hist"
        ))
    ])
    return model

def train_and_save_model(X_train, y_train, save_path="posting_model.pkl"):
    model = build_pipeline()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ Model trained and saved to {save_path}")

def load_model(path="post_scheduler/posting_model.pkl"):
    return joblib.load(path)
