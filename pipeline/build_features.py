import pandas as pd
from pathlib import Path
from collections import deque

def build_features(games):
    out_path = Path("data/processed/features_v2a.parquet")
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    team_state = {}
    rows = []

    def get_state(team_id):
        if team_id not in team_state:
            team_state[team_id] = {
                "games": 0, "wins": 0, "runs_for": 0, "runs_against": 0,
                "recent": deque(maxlen=10)
            }
        return team_state[team_id]

    def recent_stats(state):
        recent = list(state["recent"])
        if not recent:
            return {"recent_win_pct": 0.5, "recent_rpg": 4.3, "recent_rapg": 4.3}
        n = len(recent)
        return {
            "recent_win_pct": sum(x[0] for x in recent) / n,
            "recent_rpg": sum(x[1] for x in recent) / n,
            "recent_rapg": sum(x[2] for x in recent) / n,
        }

    for _, g in games.iterrows():
        home_id, away_id = g["home_team_id"], g["away_team_id"]
        hs, aw = get_state(home_id), get_state(away_id)

        if hs["games"] >= 10 and aw["games"] >= 10:
            home_win_pct = hs["wins"] / hs["games"]
            away_win_pct = aw["wins"] / aw["games"]
            home_rpg = hs["runs_for"] / hs["games"]
            away_rpg = aw["runs_for"] / aw["games"]
            home_rapg = hs["runs_against"] / hs["games"]
            away_rapg = aw["runs_against"] / aw["games"]
            home_rd = home_rpg - home_rapg
            away_rd = away_rpg - away_rapg
            hr = recent_stats(hs)
            ar = recent_stats(aw)

            rows.append({
                "game_pk": g["game_pk"],
                "date": g["date"],
                "season": g["season"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "win_pct_diff": home_win_pct - away_win_pct,
                "home_rpg": home_rpg,
                "away_rpg": away_rpg,
                "rpg_diff": home_rpg - away_rpg,
                "home_rapg": home_rapg,
                "away_rapg": away_rapg,
                "rapg_diff": away_rapg - home_rapg,
                "home_run_diff_pg": home_rd,
                "away_run_diff_pg": away_rd,
                "run_diff_per_game_diff": home_rd - away_rd,
                "home_recent_win_pct": hr["recent_win_pct"],
                "away_recent_win_pct": ar["recent_win_pct"],
                "recent_win_pct_diff": hr["recent_win_pct"] - ar["recent_win_pct"],
                "home_recent_rpg": hr["recent_rpg"],
                "away_recent_rpg": ar["recent_rpg"],
                "recent_rpg_diff": hr["recent_rpg"] - ar["recent_rpg"],
                "home_recent_rapg": hr["recent_rapg"],
                "away_recent_rapg": ar["recent_rapg"],
                "recent_rapg_diff": ar["recent_rapg"] - hr["recent_rapg"],
                "home_field": 1,
                "home_win": g["home_win"],
            })

        home_win = int(g["home_win"] == 1)
        away_win = 1 - home_win
        hs["games"] += 1
        hs["wins"] += home_win
        hs["runs_for"] += g["home_score"]
        hs["runs_against"] += g["away_score"]
        hs["recent"].append((home_win, g["home_score"], g["away_score"]))
        aw["games"] += 1
        aw["wins"] += away_win
        aw["runs_for"] += g["away_score"]
        aw["runs_against"] += g["home_score"]
        aw["recent"].append((away_win, g["away_score"], g["home_score"]))

    features = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"Saved {len(features)} feature rows to {out_path}")
    return features
