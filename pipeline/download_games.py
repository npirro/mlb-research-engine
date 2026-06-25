import requests
import pandas as pd
from pathlib import Path
from datetime import date

MLB_BASE = "https://statsapi.mlb.com/api/v1"

def fetch_schedule_for_date(day):
    url = f"{MLB_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": day,
        "hydrate": "team,linescore"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def download_games(start_season=2023, end_season=2025):
    """
    Downloads final MLB regular-season games using MLB Stats API.

    Output:
    data/raw/games_YYYY_YYYY.parquet
    """
    out_path = Path(f"data/raw/games_{start_season}_{end_season}.parquet")
    if out_path.exists():
        print(f"Using cached file: {out_path}")
        return pd.read_parquet(out_path)

    all_rows = []

    for season in range(start_season, end_season + 1):
        print(f"Downloading season {season}...")

        # Broad regular-season date range. MLB API returns games only on valid dates.
        dates = pd.date_range(f"{season}-03-20", f"{season}-10-05", freq="D")

        for d in dates:
            day = d.date().isoformat()
            try:
                data = fetch_schedule_for_date(day)
            except Exception as e:
                print(f"Skipping {day}: {e}")
                continue

            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    status = game.get("status", {}).get("detailedState", "")
                    game_type = game.get("gameType", "")

                    if game_type != "R":
                        continue

                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})

                    home_score = home.get("score")
                    away_score = away.get("score")

                    if home_score is None or away_score is None:
                        continue

                    if status not in ["Final", "Game Over", "Completed Early"]:
                        continue

                    home_team = home.get("team", {})
                    away_team = away.get("team", {})

                    row = {
                        "game_pk": game.get("gamePk"),
                        "date": day,
                        "season": season,
                        "home_team_id": home_team.get("id"),
                        "home_team": home_team.get("abbreviation") or home_team.get("teamName") or home_team.get("name"),
                        "away_team_id": away_team.get("id"),
                        "away_team": away_team.get("abbreviation") or away_team.get("teamName") or away_team.get("name"),
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_win": 1 if home_score > away_score else 0,
                    }
                    all_rows.append(row)

    df = pd.DataFrame(all_rows).drop_duplicates("game_pk")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df)} games to {out_path}")
    return df
