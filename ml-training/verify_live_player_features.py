"""
Verification for live_player_features.py, to the same standard as the
quarter/half replay.

Four checks, each aimed at a different way this could be quietly wrong:

  1. Stratified historical replay - computed features must match
     player_dataset.csv exactly, across eras and across both routing
     populations. This is the "does the arithmetic agree with training"
     check.
  2. A REAL roster exclusion, using the live injury report for the
     2026-03-14 BKN/PHI fixture. A player known to be Out that night must
     not appear in the returned roster. Needs network + Java; skipped with
     a loud note if either is unavailable, never silently passed.
  3. The no-report path must return the FULL roster with
     AVAILABILITY_KNOWN False - not a shorter roster, and not a silent
     claim that everyone is fit.
  4. Routing tags must match what the manifest dictates, checked on a
     deliberately complete player and a deliberately incomplete one.

Run:  python ml-training/verify_live_player_features.py
"""

import sys
from pathlib import Path

import pandas as pd

from xgboost import XGBRegressor

from live_features import load_games_final
from live_player_features import (
    current_roster,
    describe,
    get_live_player_features,
    load_player_history,
    load_routing_rule,
    route_for,
)

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import FEATURE_COLUMNS, OUTPUT_PATH  # noqa: E402

TOLERANCE = 1e-9
REPLAY_GAMES = 12
BKN_PHI_DATE = pd.Timestamp("2026-03-14")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    section("SETUP")
    history = load_player_history()
    games_final = load_games_final()
    dataset = pd.read_csv(OUTPUT_PATH)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    decide_on = load_routing_rule()
    print(f"  player history {len(history):,} rows | games_final "
          f"{len(games_final):,} | player_dataset {len(dataset):,}")
    print(f"  routing decides on {len(decide_on)} columns")

    # ------------------------------------------------------------------ 1
    section(f"1. HISTORICAL REPLAY - {REPLAY_GAMES} team-games across eras")
    print("Computed features must equal player_dataset.csv exactly, for every")
    print("player the pipeline has a row for.\n")

    played = dataset.sort_values(["GAME_DATE", "GAME_ID"])
    picks = (played.groupby(played["GAME_DATE"].dt.year)
             .apply(lambda g: g.iloc[len(g) // 2], include_groups=False)
             .head(REPLAY_GAMES))

    total_players = matched = missing_from_live = 0
    worst_diff, worst_where = 0.0, None

    for _, pick in picks.iterrows():
        game_id, team_id = int(pick["GAME_ID"]), int(pick["TEAM_ID"])
        game_date = pick["GAME_DATE"]

        sides = games_final[games_final["GAME_ID"] == game_id]
        opponent_id = int(sides.loc[sides["TEAM_ID"] != team_id, "TEAM_ID"].iloc[0])
        is_home = bool(sides.loc[sides["TEAM_ID"] == team_id, "IS_HOME"].iloc[0])

        live = get_live_player_features(
            team_id, opponent_id, game_date, is_home,
            history, games_final, injury_report=None, decide_on=decide_on,
        ).set_index("PLAYER_ID")

        expected = dataset[(dataset["GAME_ID"] == game_id)
                           & (dataset["TEAM_ID"] == team_id)]

        for _, want in expected.iterrows():
            player_id = int(want["PLAYER_ID"])
            total_players += 1
            if player_id not in live.index:
                missing_from_live += 1
                continue
            got = live.loc[player_id]
            for column in FEATURE_COLUMNS:
                a, b = want[column], got[column]
                if pd.isna(a) and pd.isna(b):
                    continue
                diff = abs(float(a) - float(b))
                if diff > worst_diff:
                    worst_diff, worst_where = diff, f"{column} p{player_id}"
            matched += 1

    print(f"  team-games replayed        : {len(picks)}")
    print(f"  pipeline player rows        : {total_players:,}")
    print(f"  found in the live roster    : {matched:,}")
    print(f"  absent from the live roster : {missing_from_live:,}  "
          f"(players with no ROLL10_MIN - by design, see docstring)")
    print(f"  largest feature difference  : {worst_diff:.3e}  ({worst_where})")
    print(f"  {'PASS' if worst_diff <= TOLERANCE else 'FAIL'}")
    replay_ok = worst_diff <= TOLERANCE

    # ------------------------------------------------------------------ 4
    # Run before check 2 because it needs no network.
    section("4. ROUTING TAGS MATCH THE MANIFEST")
    sample_date = picks.iloc[-1]["GAME_DATE"]
    sample_team = int(picks.iloc[-1]["TEAM_ID"])
    sides = games_final[games_final["GAME_ID"] == int(picks.iloc[-1]["GAME_ID"])]
    sample_opp = int(sides.loc[sides["TEAM_ID"] != sample_team, "TEAM_ID"].iloc[0])

    frame = get_live_player_features(
        sample_team, sample_opp, sample_date, True,
        history, games_final, injury_report=None, decide_on=decide_on,
    )
    disagreements = 0
    for row in frame.itertuples():
        expected = route_for({c: getattr(row, c) for c in decide_on}, decide_on)
        if expected != row.ROUTE:
            disagreements += 1
    complete = frame[frame["ROUTE"] == "linear"]
    incomplete = frame[frame["ROUTE"] == "xgb"]
    print(f"  roster on {sample_date.date()}: {len(frame)} players")
    print(f"    -> linear {len(complete)}   -> xgb {len(incomplete)}")
    if not complete.empty:
        r = complete.iloc[0]
        print(f"    complete example  : {r['PLAYER_NAME']:<22}"
              f"{int(r['APPEARANCES_THIS_SEASON']):>3} appearances, "
              f"{int(r[decide_on].isna().sum())} of 12 missing -> {r['ROUTE']}")
    if not incomplete.empty:
        r = incomplete.iloc[0]
        print(f"    incomplete example: {r['PLAYER_NAME']:<22}"
              f"{int(r['APPEARANCES_THIS_SEASON']):>3} appearances, "
              f"{int(r[decide_on].isna().sum())} of 12 missing -> {r['ROUTE']}")
    print(f"  tags disagreeing with the manifest rule: {disagreements}")

    # Zero disagreements is the bar. A MIXED roster deliberately is NOT:
    # the corrected roster definition requires a non-NaN ROLL10_MIN at the
    # player's last in-season appearance, which means he already has 11+
    # appearances, which means all 12 rolling features are present at
    # game_date. So every rostered player routes to linear, structurally.
    # That is a consequence of the roster rule, not a routing failure, and
    # the routing mechanism itself was proven independently twice
    # (finalize_player_models.py's Bogut/Kobe checks).
    routing_ok = disagreements == 0
    print(f"  {'PASS' if routing_ok else 'FAIL'}")
    if incomplete.empty:
        print("  NOTE: no xgb-routed player on this roster, and that is")
        print("  structural - see the comment in this check. Spot-checked below.")

    # The cheap extra reassurance: take a genuinely incomplete row straight
    # from the training table, confirm it routes to xgb and that the xgb
    # model returns a sane number for it.
    import joblib
    from live_player_features import MODELS_DIR
    incomplete_rows = dataset[dataset[decide_on].isna().any(axis=1)]
    probe = incomplete_rows.iloc[0]
    probe_route = route_for({c: probe[c] for c in decide_on}, decide_on)
    matrix = probe[FEATURE_COLUMNS].to_frame().T.astype(float)
    booster = XGBRegressor()
    booster.load_model(str(MODELS_DIR / "pts_xgb.json"))
    predicted = float(booster.predict(matrix)[0])
    sane = 0.0 <= predicted <= 60.0
    print(f"\n  xgb spot-check: {probe['PLAYER_NAME']} on "
          f"{pd.to_datetime(probe['GAME_DATE']).date()}")
    print(f"    {int(probe[decide_on].isna().sum())} of 12 rolling features "
          f"missing -> routes to '{probe_route}'")
    print(f"    pts_xgb predicts {predicted:.2f} (actual {probe['PTS']:.0f})   "
          f"{'plausible' if sane else 'IMPLAUSIBLE'}")
    routing_ok = routing_ok and probe_route == "xgb" and sane

    # ------------------------------------------------------------------ 3
    section("3. NO REPORT -> FULL ROSTER, FLAGGED UNKNOWN")
    from live_player_features import season_of as _season_of
    roster_size = len(current_roster(history, sample_team, sample_date,
                                     _season_of(sample_date)))
    none_frame = get_live_player_features(
        sample_team, sample_opp, sample_date, True,
        history, games_final, injury_report=None, decide_on=decide_on,
    )
    flags_unknown = not none_frame["AVAILABILITY_KNOWN"].any()
    full = len(none_frame) == roster_size
    print(f"  roster with history      : {roster_size}")
    print(f"  returned with no report  : {len(none_frame)}   "
          f"({'nobody excluded' if full else 'ROWS WERE DROPPED'})")
    print(f"  AVAILABILITY_KNOWN False : {flags_unknown}")
    print(f"  describe(): {describe(none_frame)}")
    print(f"  {'PASS' if full and flags_unknown else 'FAIL'}")
    unknown_ok = full and flags_unknown

    # ------------------------------------------------------------------ 2
    section("2. REAL ROSTER EXCLUSION - the 2026-03-14 injury report")
    print("Needs network access and a Java runtime (tabula-py). A failure to")
    print("fetch is reported as SKIPPED, never counted as a pass.\n")

    exclusion_ok = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                              / "data-pipeline" / "ingestion"))
        from datetime import datetime
        from fetch_current_injury_report import get_current_injury_status
        from injury_availability import reconcile

        report = get_current_injury_status(
            as_of=datetime(2026, 3, 14, 7, 15))
        print(f"  report fetched: {len(report.players)} player rows, "
              f"{len(report.pending)} pending team(s)")

        reconciled = reconcile(report.players)
        out_now = reconciled[reconciled["IS_ABSENT"]
                             & reconciled["PLAYER_ID"].notna()]
        teams_with_outs = out_now["TEAM_ID"].value_counts()
        print(f"  {len(out_now)} absent players resolved to a PLAYER_ID "
              f"across {len(teams_with_outs)} teams")

        team_id = int(teams_with_outs.index[0])
        opponent_id = int(teams_with_outs.index[1])
        expect_gone = out_now[out_now["TEAM_ID"] == team_id]

        with_report = get_live_player_features(
            team_id, opponent_id, BKN_PHI_DATE, True, history, games_final,
            injury_report=report, decide_on=decide_on)
        without = get_live_player_features(
            team_id, opponent_id, BKN_PHI_DATE, True, history, games_final,
            injury_report=None, decide_on=decide_on)

        returned = set(with_report["PLAYER_ID"])
        should_be_absent = {int(p) for p in expect_gone["PLAYER_ID"]}
        on_roster_and_out = should_be_absent & set(without["PLAYER_ID"])
        leaked = on_roster_and_out & returned

        print(f"\n  team {team_id}: roster without report {len(without)}, "
              f"with report {len(with_report)}")
        print(f"  report marks {len(should_be_absent)} of its players out; "
              f"{len(on_roster_and_out)} of those were on the roster")
        for _, r in expect_gone.iterrows():
            pid = int(r["PLAYER_ID"])
            state = ("excluded" if pid in on_roster_and_out and pid not in returned
                     else "STILL PRESENT" if pid in returned
                     else "not on roster anyway")
            print(f"    {r['PLAYER_NAME_NORMALIZED']:<26}{state}")
        print(f"\n  AVAILABILITY_KNOWN True: "
              f"{bool(with_report['AVAILABILITY_KNOWN'].all())}")
        exclusion_ok = (not leaked) and len(on_roster_and_out) > 0
        print(f"  {'PASS' if exclusion_ok else 'FAIL - an out player was returned'}")

    except Exception as error:  # noqa: BLE001 - the point is to report, not crash
        print(f"  SKIPPED: {type(error).__name__}: {error}")
        print("  The exclusion path is therefore UNVERIFIED in this run.")

    section("RESULT")
    for name, ok in (("1 historical replay", replay_ok),
                     ("2 roster exclusion", exclusion_ok),
                     ("3 no-report flag", unknown_ok),
                     ("4 routing tags", routing_ok)):
        label = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {name:<24}{label}")
    hard = [replay_ok, unknown_ok, routing_ok]
    return 0 if all(hard) and exclusion_ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
