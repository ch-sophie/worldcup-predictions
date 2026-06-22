import sqlite3
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
DB_PATH = "world_cup_2026.db"
conn = sqlite3.connect(DB_PATH)

print("=" * 55)
print("STEP 1 — Loading raw tables from SQLite")
print("=" * 55)

results       = pd.read_sql("SELECT * FROM results",          conn)
wc_matches    = pd.read_sql("SELECT * FROM wc_all_matches",   conn)
fixtures      = pd.read_sql("SELECT * FROM fixtures_2026",    conn)
teams         = pd.read_sql("SELECT * FROM teams_2026",       conn)
elo           = pd.read_sql("SELECT * FROM elo_ratings_2026", conn)

# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

# Broad name map — extend this as mismatches
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
    """
    Adds a 'result' column: 'home_win' / 'draw' / 'away_win'.
    Rows where either score is NaN are left as NaN.
    """
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

# ─────────────────────────────────────────────
# TABLE 1 — results (all internationals 1872–2026)
# ─────────────────────────────────────────────
print("\n[1/5] Cleaning: results")

results["date"]       = pd.to_datetime(results["date"], errors="coerce")
results["home_team"]  = normalize_team(results["home_team"])
results["away_team"]  = normalize_team(results["away_team"])
results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
results["neutral"]    = results["neutral"].astype(int)

# Drop rows missing the core info
results = results.dropna(subset=["date", "home_team", "away_team"])
results = results.drop_duplicates()

# Add result label
results = add_result_label(results, "home_score", "away_score")

# Handy derived columns
results["goal_diff"]      = results["home_score"] - results["away_score"]
results["total_goals"]    = results["home_score"] + results["away_score"]
results["is_world_cup"]   = results["tournament"].str.contains(
                                "FIFA World Cup", case=False, na=False
                            ).astype(int)

print(f"  ✅ {len(results):,} rows | date range: {results['date'].min().date()} → {results['date'].max().date()}")

# ─────────────────────────────────────────────
# TABLE 2 — wc_all_matches (WC editions 1930–2022)
# ─────────────────────────────────────────────
print("\n[2/5] Cleaning: wc_all_matches")

wc_matches["date"]   = pd.to_datetime(wc_matches["date"], errors="coerce")
wc_matches["team1"]  = normalize_team(wc_matches["team1"])
wc_matches["team2"]  = normalize_team(wc_matches["team2"])
wc_matches["score1"] = pd.to_numeric(wc_matches["score1"], errors="coerce")
wc_matches["score2"] = pd.to_numeric(wc_matches["score2"], errors="coerce")
wc_matches["year"]   = pd.to_numeric(wc_matches["year"],   errors="coerce").astype("Int64")

wc_matches = wc_matches.dropna(subset=["date", "team1", "team2"])
wc_matches = wc_matches.drop_duplicates()

# Normalise stage labels
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
wc_matches["stage"] = (
    wc_matches["stage"].str.lower().str.strip().replace(stage_map)
)

wc_matches = add_result_label(wc_matches, "score1", "score2")
wc_matches["goal_diff"]   = wc_matches["score1"] - wc_matches["score2"]
wc_matches["total_goals"] = wc_matches["score1"] + wc_matches["score2"]

print(f"  ✅ {len(wc_matches):,} rows | years: {wc_matches['year'].min()}–{wc_matches['year'].max()}")

# ─────────────────────────────────────────────
# TABLE 3 — fixtures_2026
# ─────────────────────────────────────────────
print("\n[3/5] Cleaning: fixtures_2026")

fixtures["date"]   = pd.to_datetime(fixtures["date"], errors="coerce")
fixtures["team1"]  = normalize_team(fixtures["team1"])
fixtures["team2"]  = normalize_team(fixtures["team2"])
fixtures["team1_fifa_rank"] = pd.to_numeric(fixtures["team1_fifa_rank"], errors="coerce")
fixtures["team2_fifa_rank"] = pd.to_numeric(fixtures["team2_fifa_rank"], errors="coerce")

# Derived: rank difference (negative = team1 is stronger)
fixtures["rank_diff"] = fixtures["team1_fifa_rank"] - fixtures["team2_fifa_rank"]

# Flag: has the match been played yet? (scores will be added by scraper later)
fixtures["score1"]  = pd.NA
fixtures["score2"]  = pd.NA
fixtures["played"]  = 0   # scraper will flip this to 1 once result is known

fixtures = fixtures.drop_duplicates(subset=["date", "team1", "team2"])

print(f"  ✅ {len(fixtures):,} fixtures | {fixtures['date'].min().date()} → {fixtures['date'].max().date()}")

# ─────────────────────────────────────────────
# TABLE 4 — teams_2026
# ─────────────────────────────────────────────
print("\n[4/5] Cleaning: teams_2026")

teams["team"]          = normalize_team(teams["team"])
teams["confederation"] = teams["confederation"].str.upper().str.strip()
teams["fifa_rank"]     = pd.to_numeric(teams["fifa_rank"], errors="coerce")

# Boolean: is this their World Cup debut?
teams["debut_2026"] = teams["debut_2026"].str.strip().str.lower().map(
    {"yes": True, "no": False}
)

teams = teams.drop_duplicates(subset=["team"])

print(f"  ✅ {len(teams):,} teams")

# ─────────────────────────────────────────────
# TABLE 5 — elo_ratings_2026
# ─────────────────────────────────────────────
print("\n[5/5] Cleaning: elo_ratings_2026")

elo["snapshot_date"] = pd.to_datetime(elo["snapshot_date"], errors="coerce")
elo["country"]       = normalize_team(elo["country"])
elo["rating"]        = pd.to_numeric(elo["rating"], errors="coerce")
elo["year"]          = pd.to_numeric(elo["year"],   errors="coerce").astype("Int64")

elo = elo.dropna(subset=["snapshot_date", "country", "rating"])
elo = elo.drop_duplicates(subset=["snapshot_date", "country"])

# Keep only the most recent Elo snapshot per country for quick lookups
elo_latest = (
    elo.sort_values("snapshot_date")
       .groupby("country", as_index=False)
       .last()
       .rename(columns={"rating": "elo_latest"})
    [["country", "elo_latest", "snapshot_date"]]
)

print(f"  ✅ {len(elo):,} Elo snapshots | {elo['snapshot_date'].min().date()} → {elo['snapshot_date'].max().date()}")

# ─────────────────────────────────────────────
# FEATURE TABLE — recent form per team
# (computed from results, used for model input)
# ─────────────────────────────────────────────
print("\n[FEATURE] Computing last-5-match form per team...")

def compute_form(df, team_col_home, team_col_away, score_home, score_away, date_col, n=5):
    """
    For every team that appears in df, compute rolling stats
    over their last n matches (points, avg goals scored/conceded).
    Returns a DataFrame indexed by team.
    """
    records = []

    # Stack home and away rows into a single long table
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

# Use the full results table (richest history)
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
# WRITE CLEANED TABLES BACK TO DB
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Writing cleaned tables to database...")
print("=" * 55)

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
    df.to_sql(table_name, con=conn, if_exists="replace", index=False)
    print(f"  ✅ {table_name} ({len(df):,} rows)")

conn.close()

print("\nDone! All cleaned tables are in world_cup_2026.db")