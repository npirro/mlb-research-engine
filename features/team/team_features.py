from collections import deque

def create_team_state():
    return {"games": 0, "wins": 0, "runs_for": 0, "runs_against": 0, "recent": deque(maxlen=10)}

def recent_team_stats(state):
    recent = list(state["recent"])
    if not recent:
        return {"recent_win_pct": 0.5, "recent_rpg": 4.3, "recent_rapg": 4.3}
    n = len(recent)
    return {
        "recent_win_pct": sum(x[0] for x in recent) / n,
        "recent_rpg": sum(x[1] for x in recent) / n,
        "recent_rapg": sum(x[2] for x in recent) / n,
    }

def build_team_features(home_state, away_state):
    home_win_pct = home_state["wins"] / home_state["games"]
    away_win_pct = away_state["wins"] / away_state["games"]

    home_rpg = home_state["runs_for"] / home_state["games"]
    away_rpg = away_state["runs_for"] / away_state["games"]

    home_rapg = home_state["runs_against"] / home_state["games"]
    away_rapg = away_state["runs_against"] / away_state["games"]

    home_run_diff_pg = home_rpg - home_rapg
    away_run_diff_pg = away_rpg - away_rapg

    home_recent = recent_team_stats(home_state)
    away_recent = recent_team_stats(away_state)

    return {
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "win_pct_diff": home_win_pct - away_win_pct,
        "home_rpg": home_rpg,
        "away_rpg": away_rpg,
        "rpg_diff": home_rpg - away_rpg,
        "home_rapg": home_rapg,
        "away_rapg": away_rapg,
        "rapg_diff": away_rapg - home_rapg,
        "home_run_diff_pg": home_run_diff_pg,
        "away_run_diff_pg": away_run_diff_pg,
        "run_diff_per_game_diff": home_run_diff_pg - away_run_diff_pg,
        "home_recent_win_pct": home_recent["recent_win_pct"],
        "away_recent_win_pct": away_recent["recent_win_pct"],
        "recent_win_pct_diff": home_recent["recent_win_pct"] - away_recent["recent_win_pct"],
        "home_recent_rpg": home_recent["recent_rpg"],
        "away_recent_rpg": away_recent["recent_rpg"],
        "recent_rpg_diff": home_recent["recent_rpg"] - away_recent["recent_rpg"],
        "home_recent_rapg": home_recent["recent_rapg"],
        "away_recent_rapg": away_recent["recent_rapg"],
        "recent_rapg_diff": away_recent["recent_rapg"] - home_recent["recent_rapg"],
    }

def update_team_state(state, win, runs_for, runs_against):
    state["games"] += 1
    state["wins"] += int(win)
    state["runs_for"] += runs_for
    state["runs_against"] += runs_against
    state["recent"].append((int(win), runs_for, runs_against))
