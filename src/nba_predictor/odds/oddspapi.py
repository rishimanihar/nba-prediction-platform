"""Import free OddsPapi NBA moneyline history into the common odds schema."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from nba_predictor.odds.historical import (
    QUOTE_COLUMNS,
    TEAM_CODES,
    iso_utc,
    match_quotes_to_games,
    utc_time,
)

BASE_URL = "https://api.oddspapi.io/v4"
NBA_TOURNAMENT_ID = 132
MONEYLINE_MARKET = "111"  # Winner including overtime: 111 home, 112 away.


class OddsPapiRequestError(RuntimeError):
    """A provider HTTP failure without the API key or request URL."""

    def __init__(self, endpoint: str, status_code: int, code: str | None = None) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.code = code
        detail = f" ({code})" if code else ""
        super().__init__(f"OddsPapi {endpoint} returned HTTP {status_code}{detail}")


def request_json(endpoint: str, params: dict[str, str | int]) -> object:
    """Request JSON without exposing the query-string API key in exceptions."""
    url = f"{BASE_URL}/{endpoint}?{urlencode(params)}"
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; nba-prediction-platform/0.1)",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            code = None
            retry_ms = None
            try:
                error = json.load(exc)
                if isinstance(error, dict) and isinstance(error.get("error"), dict):
                    error = error["error"]
                candidate = error.get("code") if isinstance(error, dict) else None
                if isinstance(candidate, str) and candidate.replace("_", "").isalnum():
                    code = candidate[:80]
                delay = error.get("retryMs") if isinstance(error, dict) else None
                if isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay >= 0:
                    retry_ms = delay
            except (ValueError, OSError):
                pass
            if exc.code == 429 and code == "RATE_LIMITED" and attempt < 2:
                fallback = 5.0 if endpoint == "historical-odds" else 2.0
                time.sleep(min(retry_ms / 1000 if retry_ms is not None else fallback, 30) + 0.2)
                continue
            raise OddsPapiRequestError(endpoint, exc.code, code) from None
        except URLError:
            raise RuntimeError(f"OddsPapi {endpoint} could not be reached") from None
    raise RuntimeError("OddsPapi request retry loop ended unexpectedly")


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1 or not float(decimal_odds) == float(decimal_odds):
        raise ValueError("Decimal odds must be greater than 1")
    return round((decimal_odds - 1) * 100) if decimal_odds >= 2 else -round(
        100 / (decimal_odds - 1)
    )


def normalize_fixture_history(fixture: dict, history: dict) -> pd.DataFrame:
    """Reconstruct simultaneous active two-way prices from line-move timestamps."""
    if history.get("fixtureId") != fixture.get("fixtureId"):
        raise ValueError("Historical odds fixture ID does not match fixture")
    if fixture.get("tournamentId") != NBA_TOURNAMENT_ID and fixture.get(
        "tournamentSlug"
    ) != "nba":
        raise ValueError("Fixture is not in the NBA tournament")
    home_name, away_name = fixture["participant1Name"], fixture["participant2Name"]
    if home_name not in TEAM_CODES or away_name not in TEAM_CODES:
        raise ValueError(f"Unrecognized NBA team in fixture {fixture['fixtureId']}")
    tip = iso_utc(fixture["startTime"])
    rows: list[dict[str, object]] = []
    for bookmaker, data in history.get("bookmakers", {}).items():
        market = data.get("markets", {}).get(MONEYLINE_MARKET, {})
        outcomes = market.get("outcomes", {})
        histories: dict[str, list[dict]] = {}
        for side, outcome_id in (("home", "111"), ("away", "112")):
            players = outcomes.get(outcome_id, {}).get("players", {})
            histories[side] = sorted(
                (
                    {**item, "at": utc_time(item["createdAt"])}
                    for item in players.get("0", [])
                    if utc_time(item["createdAt"]) < utc_time(tip)
                ),
                key=lambda item: item["at"],
            )
        if not all(histories.values()):
            continue
        updates = sorted({item["at"] for values in histories.values() for item in values})
        latest: dict[str, dict | None] = {"home": None, "away": None}
        positions = {"home": 0, "away": 0}
        for at in updates:
            for side in ("home", "away"):
                values = histories[side]
                while positions[side] < len(values) and values[positions[side]]["at"] <= at:
                    latest[side] = values[positions[side]]
                    positions[side] += 1
            home, away = latest["home"], latest["away"]
            if not home or not away or not home.get("active") or not away.get("active"):
                continue
            home_decimal, away_decimal = float(home["price"]), float(away["price"])
            rows.append({
                "provider_event_id": fixture["fixtureId"],
                "snapshot_at": iso_utc(at.isoformat()),
                "market_updated_at": iso_utc(max(home["at"], away["at"]).isoformat()),
                "commence_time": tip,
                "home_team": TEAM_CODES[home_name], "away_team": TEAM_CODES[away_name],
                "bookmaker_key": bookmaker, "bookmaker_title": bookmaker,
                "home_moneyline": decimal_to_american(home_decimal),
                "away_moneyline": decimal_to_american(away_decimal),
                "home_decimal_odds": home_decimal, "away_decimal_odds": away_decimal,
            })
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS + ["home_decimal_odds", "away_decimal_odds"])


def ingest_day(
    day: str, bookmakers: list[str], games_path: Path, output_path: Path, api_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover one UTC day's NBA fixtures and fetch full histories for chosen books."""
    if not 1 <= len(bookmakers) <= 3:
        raise ValueError("Choose one to three bookmaker slugs")
    start = pd.Timestamp(day).date()
    end = start + timedelta(days=1)
    try:
        fixtures = request_json("fixtures", {
            "apiKey": api_key, "tournamentId": NBA_TOURNAMENT_ID,
            "from": start.isoformat(), "to": end.isoformat(),
        })
    except OddsPapiRequestError as exc:
        if exc.status_code != 404 or exc.code != "FIXTURE_NOT_FOUND":
            raise
        fixtures = []
    if not isinstance(fixtures, list):
        raise TypeError("OddsPapi fixtures response is not a list")
    nba_fixtures = [
        fixture for fixture in fixtures
        if fixture.get("tournamentId") == NBA_TOURNAMENT_ID
        or fixture.get("tournamentSlug") == "nba"
    ]
    print(f"OddsPapi returned {len(fixtures)} fixtures; {len(nba_fixtures)} NBA fixtures")
    if not nba_fixtures:
        time.sleep(2.1)  # Observe the fixtures endpoint cooldown before the next batch date.
    frames: list[pd.DataFrame] = []
    missing_history: list[dict[str, object]] = []
    for fixture in nba_fixtures:
        fixture_id = fixture["fixtureId"]
        print(f"Fetching historical odds for {fixture_id}")
        try:
            history = request_json("historical-odds", {
                "apiKey": api_key, "fixtureId": fixture_id,
                "bookmakers": ",".join(bookmakers),
            })
        except OddsPapiRequestError as exc:
            if exc.status_code != 404 or exc.code != "NOT_FOUND":
                raise
            missing_history.append({
                "fixture_id": fixture_id,
                "start_time": fixture.get("startTime"),
                "home_team": fixture.get("participant1Name"),
                "away_team": fixture.get("participant2Name"),
                "status_code": exc.status_code,
                "provider_code": exc.code,
            })
            print(f"No historical odds for {fixture_id}: HTTP 404")
            time.sleep(5.1)
            continue
        frames.append(normalize_fixture_history(fixture, history))
        time.sleep(5.1)  # Historical endpoint's documented 5-second cooldown.
    columns = QUOTE_COLUMNS + ["home_decimal_odds", "away_decimal_odds"]
    quotes = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=columns)
    print(f"Reconstructed {len(quotes)} pregame two-way moneyline quotes")
    games = pd.read_csv(games_path, dtype={"game_id": str})
    matched, unmatched = match_quotes_to_games(quotes, games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_path, index=False)
    unmatched.to_csv(output_path.with_name(output_path.stem + "_unmatched.csv"), index=False)
    missing_path = output_path.with_name(output_path.stem + "_missing_history.csv")
    pd.DataFrame(missing_history, columns=[
        "fixture_id", "start_time", "home_team", "away_team", "status_code", "provider_code",
    ]).to_csv(missing_path, index=False)
    if missing_history:
        print(f"Recorded {len(missing_history)} fixtures without historical odds in {missing_path}")
    return matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="UTC fixture date, YYYY-MM-DD")
    parser.add_argument("--check-key", action="store_true", help="Verify key with OddsPapi account")
    parser.add_argument("--bookmakers", default="fanduel,draftkings,pinnacle")
    parser.add_argument("--games", type=Path, default=Path("data/raw/games.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/oddspapi_moneylines.csv"))
    args = parser.parse_args()
    api_key = os.getenv("ODDSPAPI_API_KEY", "").strip()
    if not api_key:
        parser.error("Set ODDSPAPI_API_KEY in this terminal first")
    if args.check_key:
        account = request_json("account", {"apiKey": api_key})
        if not isinstance(account, dict) or "api_key" not in account:
            raise RuntimeError("OddsPapi did not return valid account details")
        print("OddsPapi accepted the key")
        return
    if not args.date:
        parser.error("Provide --date to ingest fixtures, or --check-key to verify access")
    matched, unmatched = ingest_day(
        args.date, [book.strip() for book in args.bookmakers.split(",")],
        args.games, args.output, api_key,
    )
    print(f"Saved {len(matched)} matched and {len(unmatched)} unmatched quotes")


if __name__ == "__main__":
    main()
