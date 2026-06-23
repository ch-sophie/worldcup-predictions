import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# SHARED HELPERS (Global Scope)
# ─────────────────────────────────────────────
NAME_MAP = {
    "usa":                     "united states",
    "us":                      "united states",
    "united states of america":"united states",
    "south korea":             "korea republic",
    "republic of korea":       "korea republic",
    "ir iran":                 "iran",
    "ivory coast":             "cote d'ivoire",
    "cape verde islands":      "cape verde",
    "trinidad & tobago":       "trinidad and tobago",
}

def normalize_team(series: pd.Series) -> pd.Series:
    """Lowercase → strip → map known variants."""
    return (
        series.astype(str)
              .str.lower()
              .str.strip()
              .replace(NAME_MAP)
    )

def add_result_label(df, home_col, away_col):
    """Adds a 'result' column: 'home_win' / 'draw' / 'away_win'."""
    conditions = [
        df[home_col] > df[away_col],
        df[home_col] == df[away_col],
        df[home_col] < df[away_col],
    ]
    choices = ["home_win", "draw", "away_win"]
    df["result"] = pd.Series(
        pd.Categorical(
            np.select(conditions, choices, default=None),
            categories=choices
        )
    )
    return df

def compute_form(df, team_col_home, team_col_away, score_home, score_away, date_col, n=5):
    """Compute rolling stats over their last n matches."""
    home_view = df[[date_col, team_col_home, team_col_away, score_home, score_away]].copy()
    home_view.columns = ["date", "team", "opponent", "gf", "ga"]
    home_view["venue"] = "home"

    away_view = df[[date_col, team_col_away, team_col_home, score_away, score_home]].copy()
    away_view.columns = ["date", "team", "opponent", "gf", "ga"]
    away_view["venue"] = "away"

    long = pd.concat([home_view, away_view], ignore_index=True)
    long = long.dropna(subset=["gf", "ga"])
    long = long.sort_values("date")

    def points(row):
        if row["gf"] > row["ga"]:   return 3
        if row["gf"] == row["ga"]:  return 1
        return 0

    long["pts"] = long.apply(points, axis=1)

    form_rows = []
    for team, grp in long.groupby("team"):
        last_n = grp.tail(n)
        form_rows.append({
            "team":             team,
            f"last{n}_pts":     last_n["pts"].sum(),
            f"last{n}_gf":      last_n["gf"].mean().round(2),
            f"last{n}_ga":      last_n["ga"].mean().round(2),
            f"last{n}_gd":      (last_n["gf"] - last_n["ga"]).mean().round(2),
            f"last{n}_wins":    (last_n["pts"] == 3).sum(),
            f"last{n}_draws":   (last_n["pts"] == 1).sum(),
            f"last{n}_losses":  (last_n["pts"] == 0).sum(),
        })
    return pd.DataFrame(form_rows)

# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION (Called by load_db.py)
# ─────────────────────────────────────────────
def run_cleaning_pipeline(db_engine, db_type="Database"):
    print("=" * 55)
    print(f"STEP 1 — Loading raw tables from {db_type}")
    print("=" * 55)

    results       = pd.read_sql("SELECT * FROM results",          db_engine)
    wc_matches    = pd.read_sql("SELECT * FROM wc_all_matches",   db_engine)
    fixtures      = pd.read_sql("SELECT * FROM fixtures_2026",    db_engine)
    teams         = pd.read_sql("SELECT * FROM teams_2026",       db_engine)
    elo           = pd.read_sql("SELECT * FROM elo_ratings_2026", db_engine)

    # ─────────────────────────────────────────────
    # TABLE 1 — results
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - 1/5] Cleaning: results")
    results["date"]       = pd.to_datetime(results["date"], errors="coerce")
    results["home_team"]  = normalize_team(results["home_team"])
    results["away_team"]  = normalize_team(results["away_team"])
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results["neutral"]    = results["neutral"].astype(int)

    results = results.dropna(subset=["date", "home_team", "away_team"])
    results = results.drop_duplicates()
    results = add_result_label(results, "home_score", "away_score")

    results["goal_diff"]      = results["home_score"] - results["away_score"]
    results["total_goals"]    = results["home_score"] + results["away_score"]
    results["is_world_cup"]   = results["tournament"].str.contains(
                                    "FIFA World Cup", case=False, na=False
                                ).astype(int)

    print(f"  ✅ rows: {len(results):,}")

    # ─────────────────────────────────────────────
    # TABLE 2 — wc_all_matches
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - 2/5] Cleaning: wc_all_matches")
    wc_matches["date"]   = pd.to_datetime(wc_matches["date"], errors="coerce")
    wc_matches["team1"]  = normalize_team(wc_matches["team1"])
    wc_matches["team2"]  = normalize_team(wc_matches["team2"])
    wc_matches["score1"] = pd.to_numeric(wc_matches["score1"], errors="coerce")
    wc_matches["score2"] = pd.to_numeric(wc_matches["score2"], errors="coerce")
    wc_matches["year"]   = pd.to_numeric(wc_matches["year"],   errors="coerce").astype("Int64")

    wc_matches = wc_matches.dropna(subset=["date", "team1", "team2"])
    wc_matches = wc_duplicates = wc_matches.drop_duplicates()

    stage_map = {
        "group stage":        "Group Stage",
        "round of 16":        "Round of 16",
        "quarterfinals":      "Quarter-finals",
        "quarter-finals":     "Quarter-finals",
        "semi-finals":        "Semi-finals",
        "semifinals":         "Semi-finals",
        "third place":        "Third Place",
        "third-place playoff":"Third Place",
        "final":              "Final",
    }
    wc_matches["stage"] = wc_matches["stage"].str.lower().str.strip().replace(stage_map)
    wc_matches = add_result_label(wc_matches, "score1", "score2")
    wc_matches["goal_diff"]   = wc_matches["score1"] - wc_matches["score2"]
    wc_matches["total_goals"] = wc_matches["score1"] + wc_matches["score2"]

    print(f"  ✅ rows: {len(wc_matches):,}")

    # ─────────────────────────────────────────────
    # TABLE 3 — fixtures_2026
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - 3/5] Cleaning: fixtures_2026")
    fixtures["date"]   = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures["team1"]  = normalize_team(fixtures["team1"])
    fixtures["team2"]  = normalize_team(fixtures["team2"])
    fixtures["team1_fifa_rank"] = pd.to_numeric(fixtures["team1_fifa_rank"], errors="coerce")
    fixtures["team2_fifa_rank"] = pd.to_numeric(fixtures["team2_fifa_rank"], errors="coerce")
    fixtures["rank_diff"] = fixtures["team1_fifa_rank"] - fixtures["team2_fifa_rank"]

    if "score1" not in fixtures.columns:
        fixtures["score1"]  = pd.NA
    if "score2" not in fixtures.columns:
        fixtures["score2"]  = pd.NA
    if "played" not in fixtures.columns:
        fixtures["played"]  = 0

    fixtures = fixtures.drop_duplicates(subset=["date", "team1", "team2"])
    print(f"  ✅ fixtures: {len(fixtures):,}")

    # ─────────────────────────────────────────────
    # TABLE 4 — teams_2026
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - 4/5] Cleaning: teams_2026")
    teams["team"]          = normalize_team(teams["team"])
    teams["confederation"] = teams["confederation"].str.upper().str.strip()
    teams["fifa_rank"]     = pd.to_numeric(teams["fifa_rank"], errors="coerce")
    teams["debut_2026"] = teams["debut_2026"].str.strip().str.lower().map(
        {"yes": True, "no": False}
    )
    teams = teams.drop_duplicates(subset=["team"])
    print(f"  ✅ teams: {len(teams):,}")

    # ─────────────────────────────────────────────
    # TABLE 5 — elo_ratings_2026
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - 5/5] Cleaning: elo_ratings_2026")
    elo["snapshot_date"] = pd.to_datetime(elo["snapshot_date"], errors="coerce")
    elo["country"]       = normalize_team(elo["country"])
    elo["rating"]        = pd.to_numeric(elo["rating"], errors="coerce")
    elo["year"]          = pd.to_numeric(elo["year"],   errors="coerce").astype("Int64")

    elo = elo.dropna(subset=["snapshot_date", "country", "rating"])
    elo = elo.drop_duplicates(subset=["snapshot_date", "country"])

    elo_latest = (
        elo.sort_values("snapshot_date")
           .groupby("country", as_index=False)
           .last()
           .rename(columns={"rating": "elo_latest"})
        [["country", "elo_latest", "snapshot_date"]]
    )
    print(f"  ✅ Elo snapshots: {len(elo):,}")

    # ─────────────────────────────────────────────
    # FEATURE TABLE — recent form per team
    # ─────────────────────────────────────────────
    print(f"\n[{db_type} - FEATURE] Computing last-5-match form per team...")
    clean_results_for_form = results.dropna(subset=["home_score", "away_score"])
    form_df = compute_form(
        clean_results_for_form,
        "home_team", "away_team",
        "home_score", "away_score",
        "date",
        n=5
    )
    print(f"  ✅ Form computed for {len(form_df):,} teams")

    # ─────────────────────────────────────────────
    # WRITE CLEANED TABLES BACK TO ACTIVE ENGINE
    # ─────────────────────────────────────────────
    print(f"\nWriting cleaned tables back to {db_type}...")
    write_map = {
        "results_clean":      results,
        "wc_matches_clean":   wc_matches,
        "fixtures_2026_clean":fixtures,
        "teams_2026_clean":   teams,
        "elo_ratings_clean":  elo,
        "elo_latest":         elo_latest,
        "team_form":          form_df,
    }

    for table_name, df in write_map.items():
        df.to_sql(table_name, con=db_engine, if_exists="replace", index=False)
        print(f"  ✅ {table_name} ({len(df):,} rows)")

    print(f"\n🎉 Cleaned tables are live inside {db_type}!")