from collections import defaultdict, deque
from pathlib import Path
import sys
import time

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.registry.feature_sets import (  # noqa: E402
    TEAM_FEATURES,
    STARTER_FEATURES,
    SCHEDULE_FEATURES,
)


MLB_BASE = "https://statsapi.mlb.com/api/v1"

RAW_PATH = Path("data/raw/games_with_starters_2023_2025.parquet")
FEATURES_PATH = Path("data/processed/features_v4_feature_factory.parquet")
LINEUP_CACHE_PATH = Path("data/raw/game_lineups_2023_2025.parquet")
LINEUP_CHECKPOINT_PATH = Path("data/raw/game_lineups_2023_2025_checkpoint.parquet")
EXPORT_DIR = Path("exports")

LINEUP_FEATURES = [
    "lineup_ops_diff",
    "lineup_obp_diff",
    "lineup_slg_diff",
    "lineup_iso_diff",
    "lineup_top4_ops_diff",
    "lineup_bottom5_ops_diff",
    "lineup_recent_tb_pg_diff",
    "lineup_bb_rate_diff",
    "lineup_k_rate_diff",
    "lineup_power_rate_diff",
]


def unique_features(features):
    seen = set()
    out = []

    for f in features:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fetch_boxscore(game_pk):
    r = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=30)
    r.raise_for_status()
    return r.json()


def parse_batting_order(value):
    try:
        if value is None:
            return None, None

        order_num = int(value)
        slot = order_num // 100

        if slot < 1 or slot > 9:
            return None, None

        return order_num, slot
    except Exception:
        return None, None


def get_batting_stat(player_obj, key, default=0):
    batting = player_obj.get("stats", {}).get("batting", {})
    return safe_int(batting.get(key, default), default)


def collect_team_batters(boxscore, game_row, side):
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})

    team_id = game_row[f"{side}_team_id"]
    team_name = game_row[f"{side}_team"]

    candidate_rows = []

    for _, player_obj in players.items():
        batting_order = player_obj.get("battingOrder")
        order_num, lineup_slot = parse_batting_order(batting_order)

        if order_num is None:
            continue

        person = player_obj.get("person", {})
        player_id = person.get("id")
        player_name = person.get("fullName")

        ab = get_batting_stat(player_obj, "atBats")
        hits = get_batting_stat(player_obj, "hits")
        doubles = get_batting_stat(player_obj, "doubles")
        triples = get_batting_stat(player_obj, "triples")
        hr = get_batting_stat(player_obj, "homeRuns")
        bb = get_batting_stat(player_obj, "baseOnBalls")
        so = get_batting_stat(player_obj, "strikeOuts")

        pa = get_batting_stat(player_obj, "plateAppearances")
        if pa <= 0:
            pa = ab + bb

        singles = max(hits - doubles - triples - hr, 0)
        total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)

        candidate_rows.append(
            {
                "game_pk": game_row["game_pk"],
                "date": game_row["date"],
                "season": game_row["season"],
                "side": side,
                "team_id": team_id,
                "team": team_name,
                "player_id": player_id,
                "player_name": player_name,
                "batting_order_num": order_num,
                "lineup_slot": lineup_slot,
                "ab": ab,
                "pa": pa,
                "hits": hits,
                "doubles": doubles,
                "triples": triples,
                "hr": hr,
                "bb": bb,
                "so": so,
                "total_bases": total_bases,
            }
        )

    if not candidate_rows:
        return []

    team_df = pd.DataFrame(candidate_rows).sort_values("batting_order_num")

    starter_keys = set()

    for slot, group in team_df.groupby("lineup_slot"):
        starter_index = group.sort_values("batting_order_num").index[0]
        starter_keys.add(starter_index)

    team_df["is_starter"] = team_df.index.map(lambda idx: 1 if idx in starter_keys else 0)

    return team_df.to_dict("records")


def download_lineup_cache(raw_games):
    if LINEUP_CACHE_PATH.exists():
        print(f"Using cached lineup file: {LINEUP_CACHE_PATH}")
        return pd.read_parquet(LINEUP_CACHE_PATH)

    LINEUP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()

    if LINEUP_CHECKPOINT_PATH.exists():
        print(f"Using lineup checkpoint: {LINEUP_CHECKPOINT_PATH}")
        existing = pd.read_parquet(LINEUP_CHECKPOINT_PATH)

    done_games = set(existing["game_pk"].unique()) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    raw_games = raw_games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    print("\nDownloading historical lineups from boxscores...")
    print("This may take several minutes the first time.")

    for i, game in raw_games.iterrows():
        game_pk = game["game_pk"]

        if game_pk in done_games:
            continue

        try:
            boxscore = fetch_boxscore(game_pk)
            rows.extend(collect_team_batters(boxscore, game, "home"))
            rows.extend(collect_team_batters(boxscore, game, "away"))

            done_games.add(game_pk)

            if len(done_games) % 250 == 0:
                checkpoint = pd.DataFrame(rows)
                checkpoint.to_parquet(LINEUP_CHECKPOINT_PATH, index=False)
                print(f"Checkpoint saved. Games processed: {len(done_games)}")

            time.sleep(0.025)

        except Exception as e:
            print(f"Lineup download failed for game {game_pk}: {e}")
            continue

    lineup_df = pd.DataFrame(rows)

    if lineup_df.empty:
        raise RuntimeError("No lineup rows were downloaded.")

    lineup_df = lineup_df.drop_duplicates(
        ["game_pk", "team_id", "player_id", "batting_order_num"]
    )

    lineup_df["date"] = pd.to_datetime(lineup_df["date"])
    lineup_df = lineup_df.sort_values(
        ["date", "game_pk", "side", "batting_order_num"]
    ).reset_index(drop=True)

    lineup_df.to_parquet(LINEUP_CACHE_PATH, index=False)
    lineup_df.to_parquet(LINEUP_CHECKPOINT_PATH, index=False)

    print(f"Saved lineup cache: {LINEUP_CACHE_PATH}")
    print(f"Rows: {len(lineup_df)}")

    return lineup_df


def create_player_state():
    return {
        "games": 0,
        "pa": 0,
        "ab": 0,
        "hits": 0,
        "bb": 0,
        "so": 0,
        "hr": 0,
        "total_bases": 0,
        "recent": deque(maxlen=14),
    }


def update_player_state(state, row):
    pa = safe_int(row.get("pa", 0))
    ab = safe_int(row.get("ab", 0))
    hits = safe_int(row.get("hits", 0))
    bb = safe_int(row.get("bb", 0))
    so = safe_int(row.get("so", 0))
    hr = safe_int(row.get("hr", 0))
    total_bases = safe_int(row.get("total_bases", 0))

    state["games"] += 1
    state["pa"] += pa
    state["ab"] += ab
    state["hits"] += hits
    state["bb"] += bb
    state["so"] += so
    state["hr"] += hr
    state["total_bases"] += total_bases

    state["recent"].append(
        {
            "pa": pa,
            "ab": ab,
            "hits": hits,
            "bb": bb,
            "so": so,
            "hr": hr,
            "total_bases": total_bases,
        }
    )


def regressed_rate(numerator, denominator, league_avg, prior_weight):
    denominator = max(float(denominator), 0.0)
    numerator = max(float(numerator), 0.0)

    return (numerator + league_avg * prior_weight) / (denominator + prior_weight)


def player_snapshot(state):
    pa = state["pa"]
    ab = state["ab"]

    obp = regressed_rate(state["hits"] + state["bb"], pa, 0.320, 120)
    slg = regressed_rate(state["total_bases"], ab, 0.410, 120)
    ops = obp + slg

    avg = regressed_rate(state["hits"], ab, 0.245, 120)
    iso = max(slg - avg, 0.0)

    bb_rate = regressed_rate(state["bb"], pa, 0.085, 120)
    k_rate = regressed_rate(state["so"], pa, 0.225, 120)
    power_rate = regressed_rate(state["hr"], pa, 0.032, 120)

    recent = list(state["recent"])

    if recent:
        recent_tb = sum(g["total_bases"] for g in recent)
        recent_games = len(recent)
        recent_tb_pg = recent_tb / max(recent_games, 1)
    else:
        recent_tb_pg = 1.35

    return {
        "ops": ops,
        "obp": obp,
        "slg": slg,
        "iso": iso,
        "bb_rate": bb_rate,
        "k_rate": k_rate,
        "power_rate": power_rate,
        "recent_tb_pg": recent_tb_pg,
    }


def aggregate_lineup(starter_rows, player_states):
    if starter_rows.empty:
        return {
            "lineup_ops": 0.730,
            "lineup_obp": 0.320,
            "lineup_slg": 0.410,
            "lineup_iso": 0.165,
            "lineup_top4_ops": 0.760,
            "lineup_bottom5_ops": 0.710,
            "lineup_recent_tb_pg": 1.35,
            "lineup_bb_rate": 0.085,
            "lineup_k_rate": 0.225,
            "lineup_power_rate": 0.032,
        }

    snapshots = []

    for _, row in starter_rows.iterrows():
        player_id = row["player_id"]
        slot = safe_int(row["lineup_slot"], 9)

        snap = player_snapshot(player_states[player_id])
        snap["slot"] = slot
        snapshots.append(snap)

    snap_df = pd.DataFrame(snapshots)

    top4 = snap_df[snap_df["slot"] <= 4]
    bottom5 = snap_df[snap_df["slot"] >= 5]

    if top4.empty:
        top4_ops = snap_df["ops"].mean()
    else:
        top4_ops = top4["ops"].mean()

    if bottom5.empty:
        bottom5_ops = snap_df["ops"].mean()
    else:
        bottom5_ops = bottom5["ops"].mean()

    return {
        "lineup_ops": snap_df["ops"].mean(),
        "lineup_obp": snap_df["obp"].mean(),
        "lineup_slg": snap_df["slg"].mean(),
        "lineup_iso": snap_df["iso"].mean(),
        "lineup_top4_ops": top4_ops,
        "lineup_bottom5_ops": bottom5_ops,
        "lineup_recent_tb_pg": snap_df["recent_tb_pg"].mean(),
        "lineup_bb_rate": snap_df["bb_rate"].mean(),
        "lineup_k_rate": snap_df["k_rate"].mean(),
        "lineup_power_rate": snap_df["power_rate"].mean(),
    }


def build_lineup_diffs(home_lineup, away_lineup):
    return {
        "lineup_ops_diff": home_lineup["lineup_ops"] - away_lineup["lineup_ops"],
        "lineup_obp_diff": home_lineup["lineup_obp"] - away_lineup["lineup_obp"],
        "lineup_slg_diff": home_lineup["lineup_slg"] - away_lineup["lineup_slg"],
        "lineup_iso_diff": home_lineup["lineup_iso"] - away_lineup["lineup_iso"],
        "lineup_top4_ops_diff": (
            home_lineup["lineup_top4_ops"] - away_lineup["lineup_top4_ops"]
        ),
        "lineup_bottom5_ops_diff": (
            home_lineup["lineup_bottom5_ops"] - away_lineup["lineup_bottom5_ops"]
        ),
        "lineup_recent_tb_pg_diff": (
            home_lineup["lineup_recent_tb_pg"] - away_lineup["lineup_recent_tb_pg"]
        ),
        "lineup_bb_rate_diff": (
            home_lineup["lineup_bb_rate"] - away_lineup["lineup_bb_rate"]
        ),

        # Positive means home lineup strikes out less.
        "lineup_k_rate_diff": (
            away_lineup["lineup_k_rate"] - home_lineup["lineup_k_rate"]
        ),

        "lineup_power_rate_diff": (
            home_lineup["lineup_power_rate"] - away_lineup["lineup_power_rate"]
        ),
    }


def build_lineup_feature_frame(raw_games, lineup_cache):
    rows = []
    player_states = defaultdict(create_player_state)
    current_season = None

    raw_games = raw_games.copy()
    raw_games["date"] = pd.to_datetime(raw_games["date"])
    raw_games = raw_games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    lineup_cache = lineup_cache.copy()
    lineup_cache["date"] = pd.to_datetime(lineup_cache["date"])

    by_game = {
        game_pk: group.copy()
        for game_pk, group in lineup_cache.groupby("game_pk")
    }

    for _, game in raw_games.iterrows():
        season = int(game["season"])

        if current_season is None or season != current_season:
            player_states = defaultdict(create_player_state)
            current_season = season

        game_pk = game["game_pk"]
        game_lineups = by_game.get(game_pk, pd.DataFrame())

        home_starters = pd.DataFrame()
        away_starters = pd.DataFrame()

        if not game_lineups.empty:
            home_starters = game_lineups[
                (game_lineups["side"] == "home") & (game_lineups["is_starter"] == 1)
            ]

            away_starters = game_lineups[
                (game_lineups["side"] == "away") & (game_lineups["is_starter"] == 1)
            ]

        home_lineup = aggregate_lineup(home_starters, player_states)
        away_lineup = aggregate_lineup(away_starters, player_states)

        rows.append(
            {
                "game_pk": game_pk,
                **build_lineup_diffs(home_lineup, away_lineup),
            }
        )

        # Update player states AFTER creating features.
        # This prevents same-game batting performance leakage.
        if not game_lineups.empty:
            for _, batter_row in game_lineups.iterrows():
                update_player_state(player_states[batter_row["player_id"]], batter_row)

    return pd.DataFrame(rows)


def evaluate_holdout(df, features, test_season=None):
    if test_season is None:
        test_season = int(df["season"].max())

    features = unique_features(features)

    data = df.dropna(subset=features + ["home_win"]).copy()

    train_df = data[data["season"] < test_season]
    test_df = data[data["season"] == test_season]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(train_df[features], train_df["home_win"])

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs >= 0.50).astype(int)

    return {
        "test_season": test_season,
        "games": len(test_df),
        "accuracy": accuracy_score(test_df["home_win"], preds),
        "log_loss": log_loss(test_df["home_win"], probs),
        "brier": brier_score_loss(test_df["home_win"], probs),
    }


def run_feature_set_comparison(df):
    feature_sets = {
        "team_only": TEAM_FEATURES + SCHEDULE_FEATURES,
        "lineup_only": LINEUP_FEATURES + SCHEDULE_FEATURES,
        "team_plus_lineup": TEAM_FEATURES + LINEUP_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter": TEAM_FEATURES + STARTER_FEATURES + SCHEDULE_FEATURES,
        "team_plus_starter_lineup": (
            TEAM_FEATURES + STARTER_FEATURES + LINEUP_FEATURES + SCHEDULE_FEATURES
        ),
    }

    rows = []

    print("\n=== Lineup Feature Set Comparison: 2025 Holdout ===")

    for name, features in feature_sets.items():
        result = evaluate_holdout(df, features)

        row = {
            "feature_set": name,
            **result,
        }

        rows.append(row)

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.3f}, "
            f"LogLoss={result['log_loss']:.4f}, "
            f"Brier={result['brier']:.4f}"
        )

    out = pd.DataFrame(rows).sort_values("log_loss")
    out.to_csv(EXPORT_DIR / "lineup_feature_set_comparison.csv", index=False)

    return out, feature_sets


def run_lineup_ablation(df):
    baseline_features = TEAM_FEATURES + SCHEDULE_FEATURES
    baseline = evaluate_holdout(df, baseline_features)

    rows = []

    print("\n=== Lineup Feature Ablation ===")
    print(
        f"baseline_team_only: "
        f"Accuracy={baseline['accuracy']:.3f}, "
        f"LogLoss={baseline['log_loss']:.4f}, "
        f"Brier={baseline['brier']:.4f}"
    )

    for feature in LINEUP_FEATURES:
        trial_features = baseline_features + [feature]
        result = evaluate_holdout(df, trial_features)

        row = {
            "feature": feature,
            **result,
            "log_loss_delta": result["log_loss"] - baseline["log_loss"],
            "brier_delta": result["brier"] - baseline["brier"],
        }

        rows.append(row)

        print(
            f"add_{feature}: "
            f"Accuracy={result['accuracy']:.3f}, "
            f"LogLoss={result['log_loss']:.4f} "
            f"({row['log_loss_delta']:+.4f}), "
            f"Brier={result['brier']:.4f} "
            f"({row['brier_delta']:+.4f})"
        )

    out = pd.DataFrame(rows).sort_values("log_loss_delta")
    out.to_csv(EXPORT_DIR / "lineup_ablation.csv", index=False)

    return out


def run_cross_validation(df, feature_sets):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    test_seasons = seasons[1:]

    rows = []

    print("\n=== Lineup Cross-Validation by Season ===")

    for name, features in feature_sets.items():
        for season in test_seasons:
            result = evaluate_holdout(df, features, test_season=season)

            row = {
                "feature_set": name,
                **result,
            }

            rows.append(row)

            print(
                f"{name} | Test {season}: "
                f"Accuracy={result['accuracy']:.3f}, "
                f"LogLoss={result['log_loss']:.4f}, "
                f"Brier={result['brier']:.4f}"
            )

    by_season = pd.DataFrame(rows)

    summary = (
        by_season.groupby("feature_set")
        .agg(
            seasons_tested=("test_season", "nunique"),
            avg_accuracy=("accuracy", "mean"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier=("brier", "mean"),
            worst_log_loss=("log_loss", "max"),
            best_log_loss=("log_loss", "min"),
        )
        .reset_index()
        .sort_values("avg_log_loss")
    )

    by_season.to_csv(
        EXPORT_DIR / "lineup_cross_validation_by_season.csv",
        index=False,
    )

    summary.to_csv(
        EXPORT_DIR / "lineup_cross_validation_summary.csv",
        index=False,
    )

    print("\n=== Lineup Cross-Validation Summary ===")
    print(summary.to_string(index=False))

    return by_season, summary


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("=== Lineup Research Lab ===")
    print("Testing whether actual batting lineup strength beats Production v1.0.")

    raw_games = pd.read_parquet(RAW_PATH)
    raw_games["date"] = pd.to_datetime(raw_games["date"])

    base_features = pd.read_parquet(FEATURES_PATH)

    lineup_cache = download_lineup_cache(raw_games)

    starters_per_team_game = (
        lineup_cache[lineup_cache["is_starter"] == 1]
        .groupby(["game_pk", "team_id"])
        .size()
    )

    print("\nLineup cache quality:")
    print(f"Lineup rows: {len(lineup_cache)}")
    print(f"Team-games with starters: {len(starters_per_team_game)}")
    print(f"Average starters per team-game: {starters_per_team_game.mean():.2f}")

    lineup_features = build_lineup_feature_frame(raw_games, lineup_cache)

    enriched = base_features.merge(
        lineup_features,
        on="game_pk",
        how="left",
    )

    enriched.to_parquet(
        EXPORT_DIR / "features_with_lineup_lab.parquet",
        index=False,
    )

    print(f"\nRaw games: {len(raw_games)}")
    print(f"Base feature rows: {len(base_features)}")
    print(f"Enriched rows: {len(enriched)}")

    print("\nLineup feature preview:")
    print(enriched[LINEUP_FEATURES].head().to_string(index=False))

    comparison, feature_sets = run_feature_set_comparison(enriched)
    run_lineup_ablation(enriched)
    run_cross_validation(enriched, feature_sets)

    print("\nSaved:")
    print("- data/raw/game_lineups_2023_2025.parquet")
    print("- exports/features_with_lineup_lab.parquet")
    print("- exports/lineup_feature_set_comparison.csv")
    print("- exports/lineup_ablation.csv")
    print("- exports/lineup_cross_validation_by_season.csv")
    print("- exports/lineup_cross_validation_summary.csv")

    print("\nDecision Rule:")
    print("Lineup only promotes if team_plus_lineup improves")
    print("avg log loss/Brier versus team_only across cross-validation.")


if __name__ == "__main__":
    main()