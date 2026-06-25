import pandas as pd
from pathlib import Path

from features.team.team_features import create_team_state, build_team_features, update_team_state
from features.starter.starter_features import (
    create_pitcher_state,
    normalize_pitcher_id,
    starter_snapshot,
    build_starter_features,
    update_pitcher_state,
)
from features.schedule.schedule_features import build_schedule_features

def build_features(games):
    out_path = Path("data/processed/features_v4_feature_factory.parquet")

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    team_state = {}
    pitcher_state = {}
    rows = []

    def get_team_state(team_id):
        if team_id not in team_state:
            team_state[team_id] = create_team_state()
        return team_state[team_id]

    def get_pitcher_state(pitcher_id):
        pitcher_id = normalize_pitcher_id(pitcher_id)
        if pitcher_id not in pitcher_state:
            pitcher_state[pitcher_id] = create_pitcher_state()
        return pitcher_state[pitcher_id]

    for _, g in games.iterrows():
        current_date = g["date"]
        home_team_state = get_team_state(g["home_team_id"])
        away_team_state = get_team_state(g["away_team_id"])
        home_sp_state = get_pitcher_state(g.get("home_starter_id"))
        away_sp_state = get_pitcher_state(g.get("away_starter_id"))

        if home_team_state["games"] >= 10 and away_team_state["games"] >= 10:
            row = {
                "game_pk": g["game_pk"],
                "date": g["date"],
                "season": g["season"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_starter_name": g.get("home_starter_name"),
                "away_starter_name": g.get("away_starter_name"),
                "home_win": g["home_win"],
            }

            row.update(build_team_features(home_team_state, away_team_state))
            row.update(build_starter_features(
                starter_snapshot(home_sp_state, current_date),
                starter_snapshot(away_sp_state, current_date),
            ))
            row.update(build_schedule_features())
            rows.append(row)

        home_win = int(g["home_win"] == 1)
        away_win = 1 - home_win

        update_team_state(home_team_state, home_win, g["home_score"], g["away_score"])
        update_team_state(away_team_state, away_win, g["away_score"], g["home_score"])

        update_pitcher_state(home_sp_state, g, "home", current_date)
        update_pitcher_state(away_sp_state, g, "away", current_date)

    features = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"Saved {len(features)} feature rows to {out_path}")
    return features
