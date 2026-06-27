from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import sys
import time

import pandas as pd
import requests


MLB_BASE = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

EXPORT_DIR = Path("exports")


MANUAL_VENUE_COORDS = {
    "yankee stadium": (40.8296, -73.9262),
    "fenway park": (42.3467, -71.0972),
    "oriole park at camden yards": (39.2840, -76.6217),
    "rogers centre": (43.6414, -79.3894),
    "comerica park": (42.3390, -83.0485),
    "progressive field": (41.4962, -81.6852),
    "target field": (44.9817, -93.2776),
    "kauffman stadium": (39.0517, -94.4803),
    "rate field": (41.8300, -87.6339),
    "guaranteed rate field": (41.8300, -87.6339),
    "pnc park": (40.4469, -80.0057),
    "great american ball park": (39.0978, -84.5066),
    "citi field": (40.7571, -73.8458),
    "wrigley field": (41.9484, -87.6553),
    "american family field": (43.0280, -87.9712),
    "busch stadium": (38.6226, -90.1928),
    "coors field": (39.7561, -104.9942),
    "chase field": (33.4455, -112.0667),
    "tropicana field": (27.7682, -82.6534),
    "loanDepot park".lower(): (25.7781, -80.2197),
    "angel stadium": (33.8003, -117.8827),
    "petco park": (32.7073, -117.1573),
    "dodger stadium": (34.0739, -118.2400),
    "oracle park": (37.7786, -122.3893),
    "truist park": (33.8908, -84.4678),
    "t-mobile park": (47.5914, -122.3325),
    "globe life field": (32.7473, -97.0842),
    "minute maid park": (29.7573, -95.3555),
    "daikin park": (29.7573, -95.3555),
    "oakland coliseum": (37.7516, -122.2005),
    "sutter health park": (38.5804, -121.5139),
    "citizens bank park": (39.9061, -75.1665),
    "nationals park": (38.8730, -77.0074),
}


def normalize_key(value):
    value = str(value or "").lower().strip()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def get_target_date():
    if len(sys.argv) >= 2:
        return datetime.strptime(sys.argv[1], "%Y-%m-%d").date()

    return datetime.now(ZoneInfo("America/New_York")).date()


def fetch_schedule(target_date):
    params = {
        "sportId": 1,
        "startDate": target_date.isoformat(),
        "endDate": target_date.isoformat(),
        "gameTypes": "R",
        "hydrate": "team,venue,linescore,probablePitcher",
    }

    r = requests.get(f"{MLB_BASE}/schedule", params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def flatten_games(schedule_json):
    games = []

    for date_block in schedule_json.get("dates", []):
        for game in date_block.get("games", []):
            if game.get("gameType") != "R":
                continue
            games.append(game)

    return games


def get_team_abbr(game, side):
    team = game.get("teams", {}).get(side, {}).get("team", {})
    return team.get("abbreviation") or team.get("teamName") or team.get("name", side.title())


def get_game_label(game):
    return f"{get_team_abbr(game, 'away')} @ {get_team_abbr(game, 'home')}"


def get_game_status(game):
    return game.get("status", {}).get("detailedState", "")


def get_game_utc_datetime(game):
    game_datetime = game.get("gameDate")

    if not game_datetime:
        return None

    try:
        return datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
    except Exception:
        return None


def get_game_time_et(game):
    dt = get_game_utc_datetime(game)

    if dt is None:
        return "TBD"

    et = dt.astimezone(ZoneInfo("America/New_York"))

    try:
        return et.strftime("%-I:%M %p")
    except Exception:
        return et.strftime("%I:%M %p").lstrip("0")


def get_venue_id(game):
    return safe_int(game.get("venue", {}).get("id"))


def get_venue_name(game):
    return game.get("venue", {}).get("name", "Unknown Venue")


def extract_coords_from_venue_obj(venue):
    possible_locations = [
        venue.get("location", {}),
        venue.get("venues", [{}])[0].get("location", {}) if venue.get("venues") else {},
    ]

    for loc in possible_locations:
        if not isinstance(loc, dict):
            continue

        possible_coord_objs = [
            loc.get("defaultCoordinates", {}),
            loc.get("coordinates", {}),
            loc,
        ]

        for coord_obj in possible_coord_objs:
            if not isinstance(coord_obj, dict):
                continue

            lat = (
                coord_obj.get("latitude")
                or coord_obj.get("lat")
                or coord_obj.get("y")
            )
            lon = (
                coord_obj.get("longitude")
                or coord_obj.get("lng")
                or coord_obj.get("lon")
                or coord_obj.get("x")
            )

            lat = safe_float(lat)
            lon = safe_float(lon)

            if lat is not None and lon is not None:
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon, "mlb_venue_coordinates"

    return None, None, None


def fetch_venue_details(venue_id):
    if not venue_id:
        return None

    urls = [
        f"{MLB_BASE}/venues/{venue_id}",
        f"{MLB_BASE}/venues?venueIds={venue_id}",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


def get_venue_coordinates(game):
    venue = game.get("venue", {})
    venue_name = get_venue_name(game)
    venue_id = get_venue_id(game)

    lat, lon, source = extract_coords_from_venue_obj(venue)

    if lat is not None and lon is not None:
        return lat, lon, source

    details = fetch_venue_details(venue_id)

    if details:
        lat, lon, source = extract_coords_from_venue_obj(details)

        if lat is not None and lon is not None:
            return lat, lon, source

    manual = MANUAL_VENUE_COORDS.get(normalize_key(venue_name))

    if manual:
        return manual[0], manual[1], "manual_venue_coordinates"

    return None, None, "missing_coordinates"


def fetch_game_feed(game_pk):
    urls = [
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


def parse_temp_from_value(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        temp = float(value)

        if -20 <= temp <= 130:
            return temp

        return None

    text = str(value)

    match = re.search(r"(-?\d{1,3})\s*(?:degrees|degree|deg|°)", text, re.I)

    if match:
        temp = float(match.group(1))

        if -20 <= temp <= 130:
            return temp

    return None


def extract_temperature_from_mlb_feed(feed):
    if not feed:
        return None, "mlb_feed_missing"

    weather = feed.get("gameData", {}).get("weather", {})

    for key in ["temp", "temperature"]:
        temp = parse_temp_from_value(weather.get(key))

        if temp is not None:
            return temp, "mlb_game_feed_weather"

    for value in weather.values():
        temp = parse_temp_from_value(value)

        if temp is not None:
            return temp, "mlb_game_feed_weather_text"

    info_items = (
        feed.get("liveData", {})
        .get("boxscore", {})
        .get("info", [])
    )

    for item in info_items:
        label = str(item.get("label", "")).lower()
        value = item.get("value", "")

        if "weather" in label:
            temp = parse_temp_from_value(value)

            if temp is not None:
                return temp, "mlb_boxscore_weather"

    return None, "mlb_temp_unavailable"


def fetch_open_meteo_hourly_temp(lat, lon, game_utc_dt):
    if game_utc_dt is None:
        return None, None, None, "missing_game_time"

    # Pull a 3-day window to avoid timezone edge cases.
    target_et_date = game_utc_dt.astimezone(ZoneInfo("America/New_York")).date()

    start_date = target_et_date - timedelta(days=1)
    end_date = target_et_date + timedelta(days=1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    r = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None, None, None, "open_meteo_missing_hourly"

    utc_offset_seconds = safe_int(data.get("utc_offset_seconds"), default=0)
    game_local_naive = (game_utc_dt + timedelta(seconds=utc_offset_seconds)).replace(tzinfo=None)

    best = None

    for t, temp in zip(times, temps):
        if temp is None:
            continue

        try:
            forecast_dt = datetime.fromisoformat(str(t))
        except Exception:
            continue

        diff_minutes = abs((forecast_dt - game_local_naive).total_seconds()) / 60.0

        if best is None or diff_minutes < best["diff_minutes"]:
            best = {
                "temp": float(temp),
                "forecast_hour": str(t),
                "diff_minutes": diff_minutes,
            }

    if best is None:
        return None, None, None, "open_meteo_no_matching_temp"

    return (
        best["temp"],
        best["forecast_hour"],
        best["diff_minutes"],
        "open_meteo_forecast",
    )


def get_pregame_temperature(game, lat, lon):
    game_pk = safe_int(game.get("gamePk"))
    game_utc_dt = get_game_utc_datetime(game)

    # First try MLB. This works well after games are underway/final.
    feed = fetch_game_feed(game_pk)
    mlb_temp, mlb_source = extract_temperature_from_mlb_feed(feed)

    if mlb_temp is not None:
        return {
            "env_temp": mlb_temp,
            "env_temp_source": mlb_source,
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    # Pregame fallback: coordinate forecast.
    try:
        forecast_temp, forecast_hour, diff_minutes, forecast_source = (
            fetch_open_meteo_hourly_temp(lat, lon, game_utc_dt)
        )

        if forecast_temp is not None:
            return {
                "env_temp": forecast_temp,
                "env_temp_source": forecast_source,
                "forecast_hour": forecast_hour,
                "forecast_hour_diff_minutes": round(diff_minutes, 1),
            }

        return {
            "env_temp": None,
            "env_temp_source": forecast_source,
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    except Exception as e:
        return {
            "env_temp": None,
            "env_temp_source": f"open_meteo_error:{type(e).__name__}",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    target_date = get_target_date()

    print("=== Candidate v1.1 Weather Source Lab ===")
    print(f"Target date: {target_date}")
    print("Goal: generate real pregame env_temp using MLB feed when available, otherwise Open-Meteo forecast.")

    print("\nFetching schedule...")

    schedule_json = fetch_schedule(target_date)
    games = flatten_games(schedule_json)

    print(f"Games loaded: {len(games)}")

    if not games:
        print("No regular-season MLB games found.")
        return

    rows = []

    print("\nGenerating weather rows...")

    for idx, game in enumerate(games, start=1):
        venue_name = get_venue_name(game)
        venue_id = get_venue_id(game)

        lat, lon, coord_source = get_venue_coordinates(game)

        if lat is not None and lon is not None:
            weather = get_pregame_temperature(game, lat, lon)
        else:
            weather = {
                "env_temp": None,
                "env_temp_source": "missing_coordinates",
                "forecast_hour": "",
                "forecast_hour_diff_minutes": "",
            }

        row = {
            "target_date": target_date.isoformat(),
            "game_pk": safe_int(game.get("gamePk")),
            "game": get_game_label(game),
            "game_time_et": get_game_time_et(game),
            "status": get_game_status(game),
            "venue_id": venue_id,
            "venue_name": venue_name,
            "latitude": lat,
            "longitude": lon,
            "coord_source": coord_source,
            **weather,
        }

        row["has_coordinates"] = pd.notna(row["latitude"]) and pd.notna(row["longitude"])
        row["has_real_env_temp"] = pd.notna(row["env_temp"])
        row["uses_open_meteo"] = str(row["env_temp_source"]).startswith("open_meteo")
        row["uses_mlb_weather"] = str(row["env_temp_source"]).startswith("mlb_")

        rows.append(row)

        print(
            f"{idx}/{len(games)} {row['game']} | "
            f"{row['venue_name']} | "
            f"temp={row['env_temp']} | "
            f"source={row['env_temp_source']}"
        )

        time.sleep(0.10)

    out = pd.DataFrame(rows)

    out_path = EXPORT_DIR / f"candidate_v1_1_weather_source_{target_date.isoformat()}.csv"
    latest_path = EXPORT_DIR / "candidate_v1_1_weather_source_latest.csv"

    out.to_csv(out_path, index=False)
    out.to_csv(latest_path, index=False)

    display_cols = [
        "game",
        "game_time_et",
        "status",
        "venue_name",
        "coord_source",
        "env_temp",
        "env_temp_source",
        "forecast_hour",
        "forecast_hour_diff_minutes",
        "has_real_env_temp",
    ]

    print("\n=== Weather Source Output ===")
    print(out[display_cols].to_string(index=False))

    total_games = len(out)
    coord_count = int(out["has_coordinates"].sum())
    real_temp_count = int(out["has_real_env_temp"].sum())
    open_meteo_count = int(out["uses_open_meteo"].sum())
    mlb_weather_count = int(out["uses_mlb_weather"].sum())
    missing_temp_count = total_games - real_temp_count

    print("\n=== Weather Feasibility Summary ===")
    print(f"Games evaluated: {total_games}")
    print(f"Venue coordinates found: {coord_count} / {total_games}")
    print(f"Real env_temp generated: {real_temp_count} / {total_games}")
    print(f"MLB weather temps used: {mlb_weather_count} / {total_games}")
    print(f"Open-Meteo forecast temps used: {open_meteo_count} / {total_games}")
    print(f"Missing env_temp: {missing_temp_count} / {total_games}")

    print("\nSaved:")
    print(f"- {out_path}")
    print(f"- {latest_path}")

    print("\n=== Weather Source Verdict ===")

    if real_temp_count == total_games and coord_count == total_games:
        print("PASS: Pregame env_temp can be generated for every game.")
        print("Next step: integrate this weather source into candidate v1.1 live model app flow.")

    elif real_temp_count == total_games:
        print("PARTIAL PASS: Temps were generated for every game, but some coordinates required fallback or were missing.")
        print("Next step: inspect coordinate sources before app integration.")

    else:
        print("FAIL: At least one game is missing env_temp.")
        print("Do not ship candidate v1.1 until weather coverage is complete.")


if __name__ == "__main__":
    main()