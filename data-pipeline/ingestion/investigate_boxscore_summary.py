"""
Throwaway: find out what BoxScoreSummaryV3's LineScore actually returns,
before any quarter/half ingestion is designed against it.

Not part of the pipeline. Same role as the traditional and advanced
box-score investigations, both of which paid for themselves - V2 was dead
mid-run, and V3 returned 0 instead of empty for absent players.

THE QUESTION THIS EXISTS TO ANSWER IS OVERTIME, and it was worth checking
the declared schema before writing a line of ingestion. nba_api declares
V2's LineScore with per-quarter columns AND ten overtime columns:

    PTS_QTR1..PTS_QTR4, PTS_OT1..PTS_OT10, PTS

but declares V3's with no overtime columns at all:

    period1Score, period2Score, period3Score, period4Score, score

If that declaration is complete, then for any overtime game the four
period columns cannot sum to `score`, and a validator built on
"quarters must equal the final" would fail on roughly 6% of games - or,
worse, an ingestion that silently trusted the sum would quietly understate
every overtime game. Three possibilities, and only a real call separates
them:

  1. V3 returns extra columns (period5Score, ...) for overtime games that
     the declared schema simply does not list.
  2. V3 returns only four periods, and overtime points are folded into
     period4Score.
  3. V3 returns only four periods and overtime points are absent entirely,
     in which case quarter data alone cannot reconstruct a final score.

Q1 and 1H markets are unaffected either way - they only need periods 1
and 2 - but which case holds decides whether the validation check is
"quarters sum to final" or something weaker, and whether full-game
reconstruction is possible at all.

FOUR GAMES, chosen deliberately rather than at random: regulation and
overtime, from the oldest and newest seasons. Overtime games were
identified offline from the raw season CSVs' MIN column - a regulation
team-game records ~240 minutes, one overtime 265, two 290. (MIN > 240
alone is not a detector: regulation games record 238-242 from rounding.)

Run:  python data-pipeline/ingestion/investigate_boxscore_summary.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2, boxscoresummaryv3

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
GAMES_FINAL_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "games_final.csv"
)

# (label, game_id) - see the module docstring for how these were chosen.
SAMPLE_GAMES = [
    ("2025-26 regulation", "0022501188"),
    ("2025-26 OVERTIME", "0022501147"),
    ("2015-16 regulation", "0021501221"),
    ("2015-16 OVERTIME", "0021501220"),
]

REQUEST_TIMEOUT_SECONDS = 30
DELAY_BETWEEN_CALLS_SECONDS = 2


def line_score(game_id: str) -> pd.DataFrame:
    """V3's LineScore frame for one game, by name not position."""
    summary = boxscoresummaryv3.BoxScoreSummaryV3(
        game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
    )
    return summary.line_score.get_data_frame()


def period_columns(frame: pd.DataFrame) -> list:
    """Every scoring-period column actually present, in order.

    Discovered from the response, not from the declared schema - the whole
    point is that the declaration may be incomplete for overtime.
    """
    return [c for c in frame.columns if "period" in c.lower() and "score" in c.lower()]


def check_v2_is_dead(game_id: str) -> None:
    print("=" * 74)
    print(f"0. Is BoxScoreSummaryV2 dead for {game_id}, like its siblings?")
    print("=" * 74)
    try:
        summary = boxscoresummaryv2.BoxScoreSummaryV2(
            game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
        )
        frame = summary.line_score.get_data_frame()
    except Exception as error:
        print(f"  RAISED {type(error).__name__}: {error}")
        print("  -> unusable. V3 only, as planned.")
        return

    if frame.empty:
        print("  EMPTY frame, no error - the same silent deprecation as")
        print("  BoxScoreTraditionalV2. V3 only, as planned.")
    else:
        print(f"  still returns {frame.shape[0]} rows x {frame.shape[1]} columns.")
        print(f"  columns: {list(frame.columns)}")
        print("  -> V2 is alive for this game; V3 is still the right target,")
        print("     but note the release notes' April 2025 cutoff.")


def inspect_game(label: str, game_id: str, finals: dict) -> dict:
    print("\n" + "=" * 74)
    print(f"{label}  -  GAME_ID {game_id}")
    print("=" * 74)

    frame = line_score(game_id)
    print(f"  LineScore shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"  columns: {list(frame.columns)}")

    periods = period_columns(frame)
    print(f"\n  scoring-period columns found: {periods}")
    print(f"  count: {len(periods)}  "
          f"({'regulation only' if len(periods) == 4 else 'MORE THAN FOUR - overtime is represented'})")

    print("\n  FULL ROWS:")
    for _, row in frame.iterrows():
        print(f"    --- {row.get('teamTricode', row.get('teamCity', '?'))} ---")
        for key, value in row.to_dict().items():
            print(f"      {key:<24} = {value!r}")

    print("\n  ARITHMETIC: do the period columns sum to the reported score?")
    total_column = "score" if "score" in frame.columns else None
    if total_column is None:
        print(f"    !! no 'score' column; available: {list(frame.columns)}")
        return {"label": label, "periods": periods, "sums_ok": None}

    sums_ok = True
    for _, row in frame.iterrows():
        parts = [pd.to_numeric(row[c], errors="coerce") for c in periods]
        summed = sum(v for v in parts if pd.notna(v))
        reported = pd.to_numeric(row[total_column], errors="coerce")
        team = row.get("teamTricode", "?")
        team_id = int(row["teamId"]) if "teamId" in row else None

        match = pd.notna(reported) and abs(summed - reported) < 1e-9
        sums_ok &= match
        print(f"    {team}: {' + '.join(str(int(v)) for v in parts if pd.notna(v))} "
              f"= {summed:.0f}  vs score {reported}  "
              f"{'OK' if match else 'MISMATCH  <-- overtime points unaccounted for'}")

        # Independent cross-check against a table already trusted for months.
        actual = finals.get((game_id, team_id))
        if actual is not None:
            agrees = abs(float(reported) - actual) < 1e-9
            print(f"       games_final.csv PTS = {actual:.0f}  "
                  f"{'agrees' if agrees else 'DISAGREES'}")

    return {"label": label, "periods": periods, "sums_ok": sums_ok}


def load_finals() -> dict:
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PTS"])
    return {
        (str(row.GAME_ID).zfill(10), int(row.TEAM_ID)): float(row.PTS)
        for row in games.itertuples()
    }


def main():
    finals = load_finals()

    check_v2_is_dead(SAMPLE_GAMES[0][1])
    time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    results = []
    for label, game_id in SAMPLE_GAMES:
        results.append(inspect_game(label, game_id, finals))
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for r in results:
        print(f"  {r['label']:<22} {len(r['periods'])} period columns   "
              f"sums to score: {r['sums_ok']}")

    print("\n  schema identical across all four games: ", end="")
    shapes = {tuple(r["periods"]) for r in results}
    print(len(shapes) == 1)
    if len(shapes) > 1:
        print("    -> the column set VARIES by game. An ingestion script must")
        print("       normalise to a fixed schema rather than trusting each")
        print("       response's own shape, exactly as the V2/V3 traditional")
        print("       box scores needed.")
        for s in shapes:
            print(f"    {list(s)}")

    overtime = [r for r in results if "OVERTIME" in r["label"]]
    if overtime and all(r["sums_ok"] is False for r in overtime):
        print("\n  OVERTIME CONFIRMED UNRECONSTRUCTABLE from period columns alone.")
        print("  Q1 and 1H markets are unaffected (periods 1 and 2 only), but a")
        print("  'quarters sum to final' validation would fail on every OT game.")
    elif overtime and all(r["sums_ok"] for r in overtime):
        print("\n  Overtime IS represented - the sum holds even for OT games, so")
        print("  the declared schema was simply incomplete.")

    print("\nDone. Five API calls, nothing written to disk.")


if __name__ == "__main__":
    main()
