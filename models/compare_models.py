FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "sp_starts_diff",
    "sp_era_diff",
    "sp_whip_diff",
    "sp_kbb_diff",
    "sp_ip_per_start_diff",
    "sp_rest_diff",
    "sp_recent_era_diff",
    "home_field",
]

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

def compare_models(df):
    df = df.copy().dropna(subset=FEATURES + ["home_win"])
    latest = df["season"].max()
    train_df = df[df["season"] < latest].copy()
    test_df = df[df["season"] == latest].copy()
    X_train, y_train = train_df[FEATURES], train_df["home_win"]
    X_test, y_test = test_df[FEATURES], test_df["home_win"]

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, random_state=42),
    }

    rows = []
    print("")
    print("=== Model Comparison ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        row = {
            "model": name,
            "test_season": int(latest),
            "games_tested": len(test_df),
            "accuracy": accuracy_score(y_test, preds),
            "log_loss": log_loss(y_test, probs),
            "brier_score": brier_score_loss(y_test, probs),
        }
        rows.append(row)
        print(f"{name}: Accuracy={row['accuracy']:.3f}, LogLoss={row['log_loss']:.3f}, Brier={row['brier_score']:.3f}")

    results = pd.DataFrame(rows).sort_values("log_loss")
    Path("exports").mkdir(exist_ok=True)
    results.to_csv("exports/model_comparison.csv", index=False)
    print("Saved model comparison to exports/model_comparison.csv")
    return results
