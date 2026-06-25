import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from features.registry.feature_sets import FEATURE_SETS

def compare_feature_sets(df):
    rows = []
    latest = df["season"].max()

    print("")
    print("=== Feature Set Comparison ===")

    for name, features in FEATURE_SETS.items():
        data = df.dropna(subset=features + ["home_win"]).copy()
        train_df = data[data["season"] < latest].copy()
        test_df = data[data["season"] == latest].copy()

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000))
        ])

        model.fit(train_df[features], train_df["home_win"])
        probs = model.predict_proba(test_df[features])[:, 1]
        preds = (probs >= 0.50).astype(int)

        row = {
            "feature_set": name,
            "test_season": int(latest),
            "games_tested": len(test_df),
            "accuracy": accuracy_score(test_df["home_win"], preds),
            "log_loss": log_loss(test_df["home_win"], probs),
            "brier_score": brier_score_loss(test_df["home_win"], probs),
            "features": ", ".join(features),
        }
        rows.append(row)

        print(f"{name}: Accuracy={row['accuracy']:.3f}, LogLoss={row['log_loss']:.3f}, Brier={row['brier_score']:.3f}")

    results = pd.DataFrame(rows).sort_values("log_loss")
    Path("exports").mkdir(exist_ok=True)
    results.to_csv("exports/feature_set_comparison.csv", index=False)
    print("Saved feature set comparison to exports/feature_set_comparison.csv")
    return results
