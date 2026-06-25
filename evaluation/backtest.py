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

from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

def bucket_probability(p):
    if p < 0.50: return "<50%"
    if p < 0.55: return "50-55%"
    if p < 0.60: return "55-60%"
    if p < 0.65: return "60-65%"
    if p < 0.70: return "65-70%"
    return "70%+"

def evaluate_model(model, test_df):
    X = test_df[FEATURES]
    y = test_df["home_win"]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.50).astype(int)

    results = test_df.copy()
    results["home_win_prob"] = probs
    results["predicted_home_win"] = preds
    results["correct"] = (results["predicted_home_win"] == results["home_win"]).astype(int)
    results["prob_bucket"] = results["home_win_prob"].apply(bucket_probability)

    print("")
    print("=== Backtest Summary ===")
    print(f"Accuracy: {accuracy_score(y, preds):.3f}")
    print(f"Log Loss: {log_loss(y, probs):.3f}")
    print(f"Brier Score: {brier_score_loss(y, probs):.3f}")

    bucket = results.groupby("prob_bucket").agg(
        games=("game_pk", "count"),
        avg_projected_prob=("home_win_prob", "mean"),
        actual_win_rate=("home_win", "mean"),
        accuracy=("correct", "mean")
    ).reset_index()

    print("")
    print("=== Calibration Buckets ===")
    print(bucket.to_string(index=False))
    results.to_csv("exports/backtest_results.csv", index=False)
    bucket.to_csv("exports/calibration_buckets.csv", index=False)
