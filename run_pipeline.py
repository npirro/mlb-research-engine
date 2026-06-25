from pipeline.download_games import download_games
from pipeline.build_features import build_features
from models.train_logistic import train_model
from models.compare_models import compare_models
from evaluation.backtest import evaluate_model
from evaluation.feature_importance import export_feature_importance

START_SEASON = 2023
END_SEASON = 2025

def main():
    print("=== MLB Research Engine v3: Starting Pitchers ===")
    print("This run downloads boxscores to identify actual starters. First run may take longer.")
    games = download_games(START_SEASON, END_SEASON)

    print("")
    print("Building feature dataset with no-lookahead starting pitcher stats...")
    features = build_features(games)

    print("")
    print("Training logistic regression model...")
    model, test_df = train_model(features)

    print("")
    print("Evaluating model...")
    evaluate_model(model, test_df)

    print("")
    print("Exporting feature importance...")
    export_feature_importance(model)

    print("")
    print("Comparing model types...")
    compare_models(features)

    print("")
    print("Done.")
    print("Review exports/feature_importance.csv and exports/model_comparison.csv")

if __name__ == "__main__":
    main()
