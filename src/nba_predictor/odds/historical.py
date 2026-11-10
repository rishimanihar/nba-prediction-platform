"""Ingest The Odds API historical NBA moneyline snapshots and match NBA games."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from nba_predictor.odds.value import american_odds_to_decimal

API_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
EASTERN = ZoneInfo("America/New_York")
TEAM_CODES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
QUOTE_COLUMNS = [
    "provider_event_id", "snapshot_at", "market_updated_at", "commence_time",
    "home_team", "away_team", "bookmaker_key", "bookmaker_title",
    "home_moneyline", "away_moneyline",
]


def utc_time(value: str) -> datetime:
    """Require an explicit timezone and return an aware UTC datetime."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp needs a timezone: {value}")
    return parsed.astimezone(UTC)


def iso_utc(value: str) -> str:
    return utc_time(value).isoformat().replace("+00:00", "Z")


def fetch_snapshot(snapshot_at: str, api_key: str, region: str = "us") -> dict:
    """Fetch one paid historical snapshot; the key never appears in error messages."""
    requested = iso_utc(snapshot_at)
    query = urlencode({
        "apiKey": api_key, "regions": region, "markets": "h2h",
        "oddsFormat": "american", "dateFormat": "iso", "date": requested,
    })
    request = Request(f"{API_URL}?{query}", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Historical odds request failed with HTTP {exc.code}") from None
    except URLError:
        raise RuntimeError("Historical odds request failed; check network access") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError("Historical odds response has no event list")
    if utc_time(payload["timestamp"]) > utc_time(requested):
        raise ValueError("Provider returned a snapshot after the requested time")
    return payload


def normalize_snapshot(payload: dict) -> pd.DataFrame:
    """Keep complete, pre-tip, two-way NBA moneylines with provenance."""
    snapshot_at = iso_utc(payload["timestamp"])
    rows: list[dict[str, object]] = []
    for event in payload["data"]:
        if event.get("sport_key") != "basketball_nba":
            continue
        home_name, away_name = event.get("home_team"), event.get("away_team")
        if home_name not in TEAM_CODES or away_name not in TEAM_CODES:
            raise ValueError(f"Unknown NBA team in odds event {event.get('id')}")
        tip = iso_utc(event["commence_time"])
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                updated = iso_utc(market["last_update"])
                if not (utc_time(updated) <= utc_time(snapshot_at) < utc_time(tip)):
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) != 2 or {o.get("name") for o in outcomes} != {
                    home_name, away_name
                }:
                    continue
                prices = {o["name"]: o["price"] for o in outcomes}
                if any(type(price) is not int for price in prices.values()):
                    continue
                american_odds_to_decimal(prices[home_name])
                american_odds_to_decimal(prices[away_name])
                rows.append({
                    "provider_event_id": str(event["id"]), "snapshot_at": snapshot_at,
                    "market_updated_at": updated, "commence_time": tip,
                    "home_team": TEAM_CODES[home_name], "away_team": TEAM_CODES[away_name],
                    "bookmaker_key": bookmaker["key"],
                    "bookmaker_title": bookmaker.get("title", bookmaker["key"]),
                    "home_moneyline": prices[home_name], "away_moneyline": prices[away_name],
                })
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS)


def match_quotes_to_games(quotes: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match unique Eastern-date/team fixtures; return matches and unmatched quotes."""
    required = {"game_id", "game_date", "home_team", "away_team"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Games missing columns: {', '.join(sorted(missing))}")
    if games["game_id"].astype(str).duplicated().any():
        raise ValueError("Games contain duplicate game IDs")
    if quotes.empty:
        return quotes.assign(game_id=pd.Series(dtype="str")), quotes.copy()
    frame = quotes.copy()
    frame["game_date"] = pd.to_datetime(frame["commence_time"], utc=True).dt.tz_convert(
        EASTERN
    ).dt.date.astype(str)
    game_keys = games.assign(game_date=pd.to_datetime(games["game_date"]).dt.date.astype(str))
    keys = ["game_date", "home_team", "away_team"]
    counts = game_keys.groupby(keys, dropna=False).size().rename("candidate_count").reset_index()
    candidates = frame.merge(counts, on=keys, how="left")
    unique_games = game_keys.merge(counts.loc[counts["candidate_count"].eq(1), keys], on=keys)
    candidates = candidates.merge(unique_games[keys + ["game_id"]], on=keys, how="left")
    matched = candidates.loc[candidates["candidate_count"].eq(1)].copy()
    unmatched = candidates.loc[candidates["candidate_count"].isna() | candidates[
        "candidate_count"
    ].gt(1)].copy()
    unmatched["match_status"] = unmatched["candidate_count"].apply(
        lambda count: "no_game" if pd.isna(count) else "ambiguous_game"
    )
    # One provider event must never be assigned to multiple NBA games, even across snapshots.
    if matched.groupby("provider_event_id")["game_id"].nunique().gt(1).any():
        raise ValueError("Provider event maps to multiple NBA games")
    return matched.drop(columns="candidate_count"), unmatched.drop(columns="candidate_count")


def ingest_snapshots(
    snapshot_times: list[str], games_path: Path, output_path: Path,
    api_key: str, region: str = "us",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch explicit snapshots, retain all quotes, and save matched/unmatched files."""
    if not snapshot_times:
        raise ValueError("Provide at least one snapshot time")
    games = pd.read_csv(games_path, dtype={"game_id": str})
    frames = [normalize_snapshot(fetch_snapshot(at, api_key, region)) for at in snapshot_times]
    quotes = pd.concat(frames, ignore_index=True).drop_duplicates()
    matched, unmatched = match_quotes_to_games(quotes, games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_path, index=False)
    unmatched.to_csv(output_path.with_name(output_path.stem + "_unmatched.csv"), index=False)
    return matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", required=True, help="ISO 8601 time with zone; repeatable")
    parser.add_argument("--games", type=Path, default=Path("data/raw/games.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/historical_moneylines.csv"))
    parser.add_argument("--region", default="us")
    args = parser.parse_args()
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        parser.error("Set ODDS_API_KEY to a historical-access The Odds API key")
    matched, unmatched = ingest_snapshots(args.snapshot, args.games, args.output, api_key, args.region)
    print(f"Saved {len(matched)} matched and {len(unmatched)} unmatched quotes")


if __name__ == "__main__":
    main()
