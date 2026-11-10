"""Collect OddsPapi NBA moneylines across UTC fixture dates with resumable daily files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from nba_predictor.odds.historical import QUOTE_COLUMNS
from nba_predictor.odds.oddspapi import ingest_day


def collect_range(
    start_date: str,
    end_date: str,
    bookmakers: list[str],
    games_path: Path,
    daily_dir: Path,
    output_path: Path,
    api_key: str,
    seed_paths: list[Path] | None = None,
) -> pd.DataFrame:
    """Save each day once and combine matched quotes, including optional earlier files."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be no later than end_date")
    if not 1 <= len(bookmakers) <= 3 or len(set(bookmakers)) != len(bookmakers):
        raise ValueError("Choose one to three distinct bookmaker slugs")
    daily_dir.mkdir(parents=True, exist_ok=True)
    files = list(seed_paths or [])
    for current in pd.date_range(start, end):
        day = current.date().isoformat()
        daily_path = daily_dir / f"{day}.csv"
        unmatched_path = daily_dir / f"{day}_unmatched.csv"
        manifest_path = daily_dir / f"{day}.json"
        expected_manifest = {
            "date": day, "bookmakers": bookmakers, "games_path": str(games_path.resolve())
        }
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != expected_manifest:
            raise ValueError(f"Saved daily data has different inputs: {manifest_path}")
        if daily_path.exists() and unmatched_path.exists() and manifest_path.exists():
            print(f"{day}: using saved daily files")
        else:
            matched, unmatched = ingest_day(
                day, bookmakers, games_path, daily_path, api_key
            )
            manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
            print(f"{day}: {len(matched)} matched; {len(unmatched)} unmatched")
        files.append(daily_path)
    frames = [pd.read_csv(path, dtype={"game_id": str}) for path in files]
    columns = QUOTE_COLUMNS + ["home_decimal_odds", "away_decimal_odds", "game_id", "game_date"]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(
        columns=columns
    )
    combined = combined.sort_values(
        ["commence_time", "provider_event_id", "bookmaker_key", "snapshot_at"],
        kind="stable",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} unique matched quotes across {combined['game_id'].nunique()} games")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="First UTC fixture date, inclusive")
    parser.add_argument("--end-date", required=True, help="Last UTC fixture date, inclusive")
    parser.add_argument("--bookmakers", default="fanduel")
    parser.add_argument("--games", type=Path, default=Path("data/raw/games_2025_26.csv"))
    parser.add_argument("--daily-dir", type=Path, default=Path("data/raw/oddspapi_days"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/oddspapi_combined.csv"))
    parser.add_argument("--include", type=Path, action="append", default=[], help="Earlier matched CSV")
    args = parser.parse_args()
    api_key = os.getenv("ODDSPAPI_API_KEY", "").strip()
    if not api_key:
        parser.error("Set ODDSPAPI_API_KEY in this terminal first")
    books = [book.strip() for book in args.bookmakers.split(",")]
    collect_range(
        args.start_date, args.end_date, books, args.games, args.daily_dir,
        args.output, api_key, args.include,
    )


if __name__ == "__main__":
    main()
