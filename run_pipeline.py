from pipeline.download_games import download_games
from pipeline.build_features import build_features
from models.train_logistic import train_model
from models.compare_models import compare_models
from evaluation.backtest import evaluate_model
from evaluation.feature_importance import export_feature_importance
from model_lab.compare_feature_sets import compare_feature_sets

START_SEASON = 2023
END_SEASON = 2025

def main():
    print("=== MLB Research Engine v4: Feature Factory ===")
    print("Numbers decide. Not opinions.")
    games = download_games(START_SEASON, END_SEASON)
    features = build_features(games)

    compare_feature_sets(features)

    model, test_df = train_model(features)
    evaluate_model(model, test_df)
    export_feature_importance(model)
    compare_models(features)

    print("")
    print("Done.")
    print("Review:")
    print("- exports/feature_set_comparison.csv")
    print("- exports/model_comparison.csv")
    print("- exports/feature_importance.csv")
    print("- exports/calibration_buckets.csv")

if __name__ == "__main__":
    main()
