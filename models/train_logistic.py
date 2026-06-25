FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "home_recent_win_pct",
    "away_recent_win_pct",
    "recent_win_pct_diff",
    "home_recent_rpg",
    "away_recent_rpg",
    "recent_rpg_diff",
    "home_field",
]

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

def train_model(df):
    df = df.copy().dropna(subset=FEATURES + ["home_win"])
    latest = df["season"].max()
    train_df = df[df["season"] < latest].copy()
    test_df = df[df["season"] == latest].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000))
    ])
    model.fit(train_df[FEATURES], train_df["home_win"])

    probs = model.predict_proba(test_df[FEATURES])[:, 1]
    preds = (probs >= 0.50).astype(int)

    print("")
    print("=== Holdout Test Results ===")
    print(f"Test Season: {latest}")
    print(f"Games Tested: {len(test_df)}")
    print(f"Accuracy: {accuracy_score(test_df['home_win'], preds):.3f}")
    print(f"Log Loss: {log_loss(test_df['home_win'], probs):.3f}")
    print(f"Brier Score: {brier_score_loss(test_df['home_win'], probs):.3f}")

    Path("exports").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "latest_test_season": int(latest)}, "exports/mlb_logistic_model.joblib")
    return model, test_df
