import os
import argparse
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from collections import defaultdict

load_dotenv()

# --- clients ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(CURRENT_DIR, "model.pkl")
DATA_DIR    = os.path.join(CURRENT_DIR, "..", "data")  # where pipeline.py downloads Kaggle files

HOSTS = {'united states', 'mexico', 'canada'}

FEATURE_COLS = [
    'elo_diff',
    'team1_last5_pts', 'team2_last5_pts', 'pts_diff',
    'team1_last5_gd',  'team2_last5_gd',  'gd_diff',
    'team1_is_host',   'team2_is_host',
]

# ──────────────────────────────────────────────
# 1. LOAD ALL KAGGLE DATA
# ──────────────────────────────────────────────

def load_kaggle_data():
    """
    Reads all three Kaggle CSVs from the local data/ folder.
    Returns:
        df_results   — full historical match results (results.csv)
        df_elo       — ELO ratings snapshot (elo_ratings_wc2026.csv)
        df_fixtures  — 2026 WC fixture list (wc_2026_fixtures.csv)
    """
    results_path  = os.path.join(DATA_DIR, "results.csv")
    elo_path      = os.path.join(DATA_DIR, "elo_ratings_wc2026.csv")
    fixtures_path = os.path.join(DATA_DIR, "wc_2026_fixtures.csv")

    for p in [results_path, elo_path, fixtures_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Run pipeline.py first to download Kaggle datasets."
            )

    # results.csv  →  date, home_team, away_team, home_score, away_score, ...
    df_results = pd.read_csv(results_path)
    df_results.columns = [c.lower().strip() for c in df_results.columns]
    df_results['home_score'] = pd.to_numeric(df_results['home_score'], errors='coerce')
    df_results['away_score'] = pd.to_numeric(df_results['away_score'], errors='coerce')
    df_results['home_team_clean'] = df_results['home_team'].str.lower().str.strip()
    df_results['away_team_clean'] = df_results['away_team'].str.lower().str.strip()

    # elo_ratings_wc2026.csv  →  year, snapshot_date, country, rank, rating, ...
    df_elo = pd.read_csv(elo_path)
    df_elo.columns = [c.lower().strip() for c in df_elo.columns]
    # Keep only the most recent snapshot per country
    df_elo = df_elo.sort_values('snapshot_date').groupby('country').last().reset_index()
    df_elo['country_clean'] = df_elo['country'].str.lower().str.strip()

    # wc_2026_fixtures.csv  →  group, stage, team1, team2, venue, city, country, date, ...
    df_fixtures = pd.read_csv(fixtures_path)
    df_fixtures.columns = [c.lower().strip() for c in df_fixtures.columns]
    df_fixtures['team1_clean'] = df_fixtures['team1'].str.lower().str.strip()
    df_fixtures['team2_clean'] = df_fixtures['team2'].str.lower().str.strip()
    df_fixtures['match_id']    = range(1, len(df_fixtures) + 1)

    print(f"✓ Loaded {len(df_results):,} historical results")
    print(f"✓ Loaded {len(df_elo)} ELO ratings")
    print(f"✓ Loaded {len(df_fixtures)} WC 2026 fixtures")

    return df_results, df_elo, df_fixtures


# ──────────────────────────────────────────────
# 2. BUILD REFERENCE DICTS FROM KAGGLE DATA
# ──────────────────────────────────────────────

def build_elo_dict(df_elo):
    """Maps team name (lowercase) → latest ELO rating."""
    return dict(zip(df_elo['country_clean'], df_elo['rating']))


def build_form_dict(df_results, n=5):
    """
    Computes last-N-match points and goal difference per team
    from historical results.csv data.
    Returns dict: team → {'last5_pts': float, 'last5_gd': float}
    """
    df = df_results[df_results['home_score'].notna()].copy()
    df = df.sort_values('date')

    form = defaultdict(lambda: {'pts': [], 'gd': []})

    for _, row in df.iterrows():
        t1, t2 = row['home_team_clean'], row['away_team_clean']
        s1, s2 = row['home_score'], row['away_score']
        gd1, gd2 = s1 - s2, s2 - s1

        if s1 > s2:
            pts1, pts2 = 3, 0
        elif s2 > s1:
            pts1, pts2 = 0, 3
        else:
            pts1, pts2 = 1, 1

        form[t1]['pts'].append(pts1)
        form[t1]['gd'].append(gd1)
        form[t2]['pts'].append(pts2)
        form[t2]['gd'].append(gd2)

    form_dict = {}
    for team, data in form.items():
        last_pts = data['pts'][-n:]
        last_gd  = data['gd'][-n:]
        form_dict[team] = {
            'last5_pts': sum(last_pts) / len(last_pts) if last_pts else 7.5,
            'last5_gd':  sum(last_gd)  / len(last_gd)  if last_gd  else 0.0,
        }

    print(f"✓ Built form dict for {len(form_dict)} teams (last {n} matches each)")
    return form_dict


# ──────────────────────────────────────────────
# 3. FEATURE BUILDING & PROBABILITY CACHE
# ──────────────────────────────────────────────

def build_features(t1, t2, elo_dict, form_dict):
    e1 = elo_dict.get(t1, 1500);  e2 = elo_dict.get(t2, 1500)
    f1 = form_dict.get(t1, {});   f2 = form_dict.get(t2, {})
    return {
        'elo_diff':         e1 - e2,
        'team1_last5_pts':  f1.get('last5_pts', 7.5),
        'team2_last5_pts':  f2.get('last5_pts', 7.5),
        'pts_diff':         f1.get('last5_pts', 7.5) - f2.get('last5_pts', 7.5),
        'team1_last5_gd':   f1.get('last5_gd', 0),
        'team2_last5_gd':   f2.get('last5_gd', 0),
        'gd_diff':          f1.get('last5_gd', 0) - f2.get('last5_gd', 0),
        'team1_is_host':    int(t1 in HOSTS),
        'team2_is_host':    int(t2 in HOSTS),
    }


def build_prob_cache(all_teams, model, elo_dict, form_dict):
    pairs = [(t1, t2) for t1 in all_teams for t2 in all_teams if t1 != t2]
    rows  = [build_features(t1, t2, elo_dict, form_dict) for t1, t2 in pairs]
    X     = pd.DataFrame(rows)[FEATURE_COLS]
    probs = model.predict_proba(X)
    cmap  = {c: i for i, c in enumerate(model.classes_)}

    cache_gs, cache_ko = {}, {}
    for (t1, t2), p in zip(pairs, probs):
        ph  = p[cmap.get(2, 0)]
        pd_ = p[cmap.get(1, 0)]
        pa  = p[cmap.get(0, 0)]
        cache_gs[(t1, t2)] = (ph, pd_, pa)
        total = ph + pa
        cache_ko[(t1, t2)] = (ph + pd_ * ph / total) if total > 0 else 0.5

    return cache_gs, cache_ko


# ──────────────────────────────────────────────
# 4. SUPABASE HELPER
# ──────────────────────────────────────────────

def upsert_to_supabase(table: str, records: list, conflict_col: str):
    batch_size = 100
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table(table).upsert(batch, on_conflict=conflict_col).execute()
        total += len(batch)
    print(f"  ✓ Upserted {total} rows to '{table}'")


# ──────────────────────────────────────────────
# 5. MATCH PREDICTIONS
# ──────────────────────────────────────────────

def run_match_predictions(model, df_fixtures, elo_dict, form_dict):
    print("\n--- Match Predictions ---")

    if df_fixtures.empty:
        print("No fixtures found in wc_2026_fixtures.csv.")
        return

    results = []
    for _, row in df_fixtures.iterrows():
        t1 = row['team1_clean']
        t2 = row['team2_clean']

        feats = build_features(t1, t2, elo_dict, form_dict)
        X     = pd.DataFrame([feats])[FEATURE_COLS]
        probs = model.predict_proba(X)[0]
        cmap  = {c: i for i, c in enumerate(model.classes_)}

        p1  = float(probs[cmap.get(2, 0)])
        pd_ = float(probs[cmap.get(1, 0)])
        p2  = float(probs[cmap.get(0, 0)])

        results.append({
            'match_id':       int(row['match_id']),
            'date':           str(row['date']),
            'stage':          str(row['stage']),
            'team1':          str(row['team1']),
            'team2':          str(row['team2']),
            'prob_team1_win': round(p1 * 100, 1),
            'prob_draw':      round(pd_ * 100, 1),
            'prob_team2_win': round(p2 * 100, 1),
            'prediction': (
                str(row['team1']) if p1 >= pd_ and p1 >= p2
                else str(row['team2']) if p2 >= pd_
                else 'Draw'
            ),
        })

        print("  -> Clearing old records from 'predictions_2026' table...")
        try:
            # In Supabase Python, filtering for any valid id or performing a gte delete cleans the table
            supabase.table('predictions_2026').delete().neq('match_id', 0).execute()
        except Exception as e:
            print(f"Warning: Could not clear old predictions ({e})")

        # Now upload ONLY the fresh rows
        upsert_to_supabase('predictions_2026', results, 'match_id')

# ──────────────────────────────────────────────
# 6. TOURNAMENT SIMULATION
# ──────────────────────────────────────────────

def run_tournament_simulation(model, df_results, df_fixtures, elo_dict, form_dict, n_sims=10_000):
    print(f"\n--- Tournament Simulation ({n_sims:,} runs) ---")

    # Build groups directly from wc_2026_fixtures.csv
    gs_fixtures = df_fixtures[df_fixtures['stage'].str.lower().str.contains('group')]
    groups = {}
    for _, row in gs_fixtures.iterrows():
        grp = row['group'].strip().upper()
        for team in [row['team1_clean'], row['team2_clean']]:
            groups.setdefault(grp, set()).add(team)
    groups = {g: list(teams) for g, teams in sorted(groups.items())}

    print(f"✓ Found {len(groups)} groups: {list(groups.keys())}")

    all_teams = [t for teams in groups.values() for t in teams]
    print(f"Pre-computing probability cache for {len(all_teams)} teams...")
    cache_gs, cache_ko = build_prob_cache(all_teams, model, elo_dict, form_dict)

    # No live scores from CSV — all group stage matches start from 0
    # (results.csv contains historical matches, not 2026 WC live scores)
    base_pts: dict = defaultdict(int)
    base_gd:  dict = defaultdict(int)
    base_gf:  dict = defaultdict(int)

    upcoming_pairs = list(zip(gs_fixtures['team1_clean'], gs_fixtures['team2_clean']))
    rands = np.random.random((n_sims, len(upcoming_pairs)))

    champion_counts  = defaultdict(int)
    finalist_counts  = defaultdict(int)
    semifinal_counts = defaultdict(int)

    print("Running simulations...")
    for sim in range(n_sims):
        pts = defaultdict(int, base_pts)
        gd  = defaultdict(int, base_gd)
        gf  = defaultdict(int, base_gf)

        for i, (t1, t2) in enumerate(upcoming_pairs):
            ph, pd_, pa = cache_gs.get((t1, t2), (0.4, 0.25, 0.35))
            r = rands[sim, i]
            if r < ph:
                pts[t1] += 3; gd[t1] += 1; gf[t1] += 1
            elif r < ph + pd_:
                pts[t1] += 1; pts[t2] += 1
            else:
                pts[t2] += 3; gd[t2] += 1; gf[t2] += 1

        standings   = {}
        third_place = []
        for grp, teams in groups.items():
            sorted_t = sorted(teams,
                key=lambda t: (pts[t], gd[t], gf[t], np.random.random()), reverse=True)
            standings[grp] = sorted_t
            if len(sorted_t) >= 3:
                t3 = sorted_t[2]
                third_place.append((pts[t3], gd[t3], gf[t3], t3))

        third_place.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best8 = [x[3] for x in third_place[:8]]

        w  = {g: standings[g][0] for g in standings}
        rv = {g: standings[g][1] for g in standings}

        # 2026 WC bracket (12 groups A–L, 16 third-place qualifiers)
        r32 = [
            (w['A'], rv['B']), (w['B'], rv['A']),
            (w['C'], rv['D']), (w['D'], rv['C']),
            (w['E'], rv['F']), (w['F'], rv['E']),
            (w['G'], rv['H']), (w['H'], rv['G']),
            (w['I'], rv['J']), (w['J'], rv['I']),
            (w['K'], rv['L']), (w['L'], rv['K']),
            (best8[0], best8[1]), (best8[2], best8[3]),
            (best8[4], best8[5]), (best8[6], best8[7]),
        ]

        def ko_round(matchups):
            return [t1 if np.random.random() < cache_ko.get((t1, t2), 0.5) else t2
                    for t1, t2 in matchups]

        r32w = ko_round(r32)
        r16w = ko_round([(r32w[i], r32w[i+1]) for i in range(0, 16, 2)])
        qfw  = ko_round([(r16w[i], r16w[i+1]) for i in range(0, 8, 2)])
        sfw  = ko_round([(qfw[i],  qfw[i+1])  for i in range(0, 4, 2)])

        for t in sfw:
            semifinal_counts[t] += 1
        finalist_counts[sfw[0]] += 1
        finalist_counts[sfw[1]] += 1
        champion_counts[ko_round([(sfw[0], sfw[1])])[0]] += 1

        if (sim + 1) % 2000 == 0:
            print(f"  {sim+1:,} / {n_sims:,} done...")

    records = []
    for rank, team in enumerate(
        sorted(all_teams, key=lambda t: champion_counts[t], reverse=True), start=1
    ):
        records.append({
            'rank':          rank,
            'team':          team.title(),
            'win_pct':       round(champion_counts[team]  / n_sims * 100, 2),
            'final_pct':     round(finalist_counts[team]  / n_sims * 100, 2),
            'semifinal_pct': round(semifinal_counts[team] / n_sims * 100, 2),
            'win_count':     champion_counts[team],
        })

    print(f"\n{'Rank':<5} {'Team':<22} {'Win%':>7} {'Final%':>8} {'Semi%':>7}")
    print("─" * 55)
    for r in records[:10]:
        medal = ["🥇", "🥈", "🥉"][r['rank'] - 1] if r['rank'] <= 3 else "  "
        print(f"{r['rank']:<5} {medal} {r['team']:<20} "
              f"{r['win_pct']:>6.1f}%  {r['final_pct']:>6.1f}%  {r['semifinal_pct']:>6.1f}%")

    upsert_to_supabase('tournament_predictions', records, 'rank')

# ──────────────────────────────────────────────
# 7. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tournament', action='store_true',
                        help='Also run tournament simulation')
    parser.add_argument('--sims', type=int, default=10_000,
                        help='Number of Monte Carlo simulations (default: 10000)')
    args = parser.parse_args()

    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print("\nLoading Kaggle datasets from data/ folder...")
    df_results, df_elo, df_fixtures = load_kaggle_data()

    try:
        response = supabase.table("fixtures_2026_upcoming").select("*").execute()
        if response.data:
            df_fixtures = pd.DataFrame(response.data)
            df_fixtures.columns = [c.lower().strip() for c in df_fixtures.columns]
            df_fixtures['team1_clean'] = df_fixtures['team1'].str.lower().str.strip()
            df_fixtures['team2_clean'] = df_fixtures['team2'].str.lower().str.strip()
            # If match_id isn't in your upcoming table schema, generate it:
            if 'match_id' not in df_fixtures.columns:
                df_fixtures['match_id'] = range(1, len(df_fixtures) + 1)
    except Exception as e:
        print(f"Warning: Could not pull from fixtures_2026_upcoming table ({e}). Using file fallback.")

    print("\nBuilding reference dicts from Kaggle data...")
    elo_dict  = build_elo_dict(df_elo)
    form_dict = build_form_dict(df_results)

    run_match_predictions(model, df_fixtures, elo_dict, form_dict)

    if args.tournament:
        run_tournament_simulation(model, df_results, df_fixtures, elo_dict, form_dict, n_sims=args.sims)

    print("\nDone.")