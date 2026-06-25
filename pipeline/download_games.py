import requests
import pandas as pd
from pathlib import Path
import time

MLB_BASE = "https://statsapi.mlb.com/api/v1"

def fetch_schedule_for_date(day):
    r = requests.get(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "date": day, "hydrate": "team,linescore"},
        timeout=20
    )
    r.raise_for_status()
    return r.json()

def fetch_boxscore(game_pk):
    r = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20)
    r.raise_for_status()
    return r.json()

def ip_to_float(ip):
    try:
        s = str(ip)
        if "." not in s:
            return float(s)
        whole, outs = s.split(".")
        return float(whole) + (float(outs) / 3.0)
    except Exception:
        return 0.0

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def blank_starter():
    return {
        "starter_id": None,
        "starter_name": None,
        "starter_ip": 0.0,
        "starter_hits": 0,
        "starter_runs": 0,
        "starter_er": 0,
        "starter_bb": 0,
        "starter_so": 0,
        "starter_hr": 0,
        "starter_pitches": 0,
    }
def blank_bullpen():
     return {
        "bullpen_ip": 0.0,
        "bullpen_hits": 0,
        "bullpen_runs": 0,
        "bullpen_er": 0,
        "bullpen_bb": 0,
        "bullpen_so": 0,
        "bullpen_hr": 0,
        "bullpen_pitches": 0,
        "bullpen_pitchers_used": 0,
    }


def find_bullpen_totals(boxscore, side):
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})

    totals = blank_bullpen()

    for _, p in players.items():
        pitching = p.get("stats", {}).get("pitching", {})

        if not pitching:
            continue

        # Skip the actual starter. Bullpen = pitchers who did NOT start.
        if safe_int(pitching.get("gamesStarted", 0)) == 1:
            continue

        ip = ip_to_float(pitching.get("inningsPitched", 0))

        if ip <= 0:
            continue

        totals["bullpen_ip"] += ip
        totals["bullpen_hits"] += safe_int(pitching.get("hits", 0))
        totals["bullpen_runs"] += safe_int(pitching.get("runs", 0))
        totals["bullpen_er"] += safe_int(pitching.get("earnedRuns", 0))
        totals["bullpen_bb"] += safe_int(pitching.get("baseOnBalls", 0))
        totals["bullpen_so"] += safe_int(pitching.get("strikeOuts", 0))
        totals["bullpen_hr"] += safe_int(pitching.get("homeRuns", 0))
        totals["bullpen_pitches"] += safe_int(pitching.get("numberOfPitches", 0))
        totals["bullpen_pitchers_used"] += 1

    return totals

def find_actual_starter(boxscore, side):
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})

    for _, p in players.items():
        pitching = p.get("stats", {}).get("pitching", {})
        if safe_int(pitching.get("gamesStarted", 0)) == 1:
            person = p.get("person", {})
            return {
                "starter_id": person.get("id"),
                "starter_name": person.get("fullName"),
                "starter_ip": ip_to_float(pitching.get("inningsPitched", 0)),
                "starter_hits": safe_int(pitching.get("hits", 0)),
                "starter_runs": safe_int(pitching.get("runs", 0)),
                "starter_er": safe_int(pitching.get("earnedRuns", 0)),
                "starter_bb": safe_int(pitching.get("baseOnBalls", 0)),
                "starter_so": safe_int(pitching.get("strikeOuts", 0)),
                "starter_hr": safe_int(pitching.get("homeRuns", 0)),
                "starter_pitches": safe_int(pitching.get("numberOfPitches", 0)),
            }

    return blank_starter()

def download_games(start_season=2023, end_season=2025):
    out_path = Path(f"data/raw/games_with_starters_{start_season}_{end_season}.parquet")
    if out_path.exists():
        print(f"Using cached file: {out_path}")
        return pd.read_parquet(out_path)

    all_rows = []

    for season in range(start_season, end_season + 1):
        print(f"Downloading season {season} with boxscores...")
        dates = pd.date_range(f"{season}-03-20", f"{season}-10-05", freq="D")

        for d in dates:
            day = d.date().isoformat()
            try:
                data = fetch_schedule_for_date(day)
            except Exception as e:
                print(f"Skipping schedule {day}: {e}")
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

                    game_pk = game.get("gamePk")

                    try:
                        boxscore = fetch_boxscore(game_pk)

                        home_starter = find_actual_starter(boxscore, "home")
                        away_starter = find_actual_starter(boxscore, "away")

                        home_bullpen = find_bullpen_totals(boxscore, "home")
                        away_bullpen = find_bullpen_totals(boxscore, "away")

                        time.sleep(0.03)

                    except Exception as e:
                        print(f"Boxscore failed for {game_pk}: {e}")
                        home_starter = blank_starter()
                        away_starter = blank_starter()
                        home_bullpen = blank_bullpen()
                        away_bullpen = blank_bullpen()

                    home_team = home.get("team", {})
                    away_team = away.get("team", {})
                    row = {
                        "game_pk": game_pk,
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

                    for k, v in home_starter.items():
                        row[f"home_{k}"] = v
                    for k, v in away_starter.items():
                        row[f"away_{k}"] = v
                    for k, v in home_bullpen.items():
                        row[f"home_{k}"] = v
                    for k, v in away_bullpen.items():
                         row[f"away_{k}"] = v

                    all_rows.append(row)

    df = pd.DataFrame(all_rows).drop_duplicates("game_pk")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df)} games with starters to {out_path}")
    return df
