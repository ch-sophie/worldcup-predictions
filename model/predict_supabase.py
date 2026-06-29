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
    raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file to update Render.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
CSV_PATH = os.path.join(CURRENT_DIR, "..", "data", "results.csv")

HOSTS        = {'united states', 'mexico', 'canada'}
FEATURE_COLS = [
    'elo_diff',
    'team1_last5_pts', 'team2_last5_pts', 'pts_diff',
    'team1_last5_gd',  'team2_last5_gd',  'gd_diff',
    'team1_is_host',   'team2_is_host',
]

# --- load data from CSV ---
def load_csv_fixtures():
    """Loads results.csv and splits it into completed (live) and upcoming fixtures."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find Kaggle dataset at: {CSV_PATH}")
        
    columns = ['date', 'team1', 'team2', 'score1', 'score2', 'tournament', 'city', 'country', 'neutral']
    df = pd.read_csv(CSV_PATH, names=columns, header=None)
    
    # Coerce scores to numbers; text strings like 'NA' naturally become NaN
    df['score1'] = pd.to_numeric(df['score1'], errors='coerce')
    df['score2'] = pd.to_numeric(df['score2'], errors='coerce')
    
    # Basic data cleaning for joining references
    df['team1_clean'] = df['team1'].astype(str).str.lower().str.strip()
    df['team2_clean'] = df['team2'].astype(str).str.lower().str.strip()
    
    # Split into completed matches vs upcoming matches
    df_live = df[df['score1'].notna()].copy()
    df_upcoming = df[df['score1'].isna()].copy()
    
    # Generate mock auto-incrementing match IDs for database tracking compatibility
    df_upcoming['match_id'] = range(1, len(df_upcoming) + 1)
    
    # Fallback to general STAGE assumption if column isn't provided by dataset
    if 'stage' not in df_upcoming.columns:
        df_upcoming['stage'] = 'GROUP_STAGE'
    if 'stage' not in df_live.columns:
        df_live['stage'] = 'GROUP_STAGE'
        
    return df_live, df_upcoming

# --- reference data from Supabase ---
def load_refs():
    """Fetches reference performance tables from Supabase."""
    elo_res = supabase.table("elo_latest").select("country, elo_latest").execute()
    elo = pd.DataFrame(elo_res.data)
    elo['country'] = elo['country'].str.lower().str.strip()
    elo_dict = dict(zip(elo['country'], elo['elo_latest']))

    form_res = supabase.table("team_form").select("*").execute()
    form = pd.DataFrame(form_res.data)
    form['team'] = form['team'].str.lower().str.strip()
    form_dict = form.set_index('team').to_dict(orient='index')

    return elo_dict, form_dict

def build_features(t1, t2, elo_dict, form_dict):
    e1 = elo_dict.get(t1, 1500); e2 = elo_dict.get(t2, 1500)
    f1 = form_dict.get(t1, {});  f2 = form_dict.get(t2, {})
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

# --- supabase helpers ---
def upsert_to_supabase(table: str, records: list, conflict_col: str):
    """Upsert records in batches of 100."""
    batch_size = 100
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table(table).upsert(batch, on_conflict=conflict_col).execute()
        total += len(batch)
    print(f"  ✓ Upserted {total} rows to '{table}'")

# --- match predictions ---
def run_match_predictions(model, df_upcoming, elo_dict, form_dict):
    print("\n--- Match Predictions ---")

    if df_upcoming.empty:
        print("No upcoming fixtures found in CSV dataset.")
        return

    results = []
    for _, row in df_upcoming.iterrows():
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

    # Print summary
    print(f"{'Date':<12} {'Team 1':<20} {'Team 2':<20} {'T1':>6} {'Draw':>6} {'T2':>6}  Prediction")
    print("─" * 82)
    for r in results:
        print(f"{r['date']:<12} {r['team1']:<20} {r['team2']:<20} "
              f"{r['prob_team1_win']:>5.1f}% {r['prob_draw']:>5.1f}% {r['prob_team2_win']:>5.1f}%"
              f"  → {r['prediction']}")

    upsert_to_supabase('predictions_2026', results, 'match_id')

# --- tournament simulation ---
def run_tournament_simulation(model, df_live, df_upcoming, elo_dict, form_dict, n_sims: int = 10_000):
    print(f"\n--- Tournament Simulation ({n_sims:,} runs) ---")

    # Load group alignment data from Supabase
    teams_res = supabase.table("teams_2026_clean").select("team, group").execute()
    df_teams = pd.DataFrame(teams_res.data)
    df_teams['team'] = df_teams['team'].str.lower().str.strip()
    groups = df_teams.groupby('group')['team'].apply(list).to_dict()

    # Filter CSV datasets to only handle the group stage fixtures for structural progression
    df_live_gs = df_live[df_live['stage'] == 'GROUP_STAGE']
    df_up_gs = df_upcoming[df_upcoming['stage'] == 'GROUP_STAGE']

    all_teams = [t for teams in groups.values() for t in teams]
    print(f"Pre-computing probability cache for {len(all_teams)} teams...")
    cache_gs, cache_ko = build_prob_cache(all_teams, model, elo_dict, form_dict)

    # Base standings built directly out of your parsed CSV values
    base_pts, base_gd, base_gf = defaultdict(int), defaultdict(int), defaultdict(int)
    for _, r in df_live_gs.iterrows():
        t1, t2 = r['team1_clean'], r['team2_clean']
        s1, s2 = int(r['score1']), int(r['score2'])
        if s1 > s2:   base_pts[t1] += 3
        elif s2 > s1: base_pts[t2] += 3
        else:         base_pts[t1] += 1; base_pts[t2] += 1
        base_gd[t1] += s1 - s2; base_gd[t2] += s2 - s1
        base_gf[t1] += s1;     base_gf[t2] += s2

    upcoming_pairs = list(zip(df_up_gs['team1_clean'], df_up_gs['team2_clean']))
    rands = np.random.random((n_sims, len(upcoming_pairs)))

    champion_counts  = defaultdict(int)
    finalist_counts  = defaultdict(int)
    semifinal_counts = defaultdict(int)

    print(f"Running simulations...")
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

        standings = {}
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

        w = {g: standings[g][0] for g in standings}
        rv = {g: standings[g][1] for g in standings}
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

        for t in sfw: semifinal_counts[t] += 1
        finalist_counts[sfw[0]] += 1
        finalist_counts[sfw[1]] += 1
        champion_counts[ko_round([(sfw[0], sfw[1])])[0]] += 1

        if (sim + 1) % 2000 == 0:
            print(f"  {sim+1:,} / {n_sims:,} done...")

    # Build and upload results
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

    # Print top 10
    print(f"\n{'Rank':<5} {'Team':<22} {'Win%':>7} {'Final%':>8} {'Semi%':>7}")
    print("─" * 55)
    for r in records[:10]:
        medal = ["🥇", "🥈", "🥉"][r['rank']-1] if r['rank'] <= 3 else "  "
        print(f"{r['rank']:<5} {medal} {r['team']:<20} "
              f"{r['win_pct']:>6.1f}%  {r['final_pct']:>6.1f}%  {r['semifinal_pct']:>6.1f}%")

    upsert_to_supabase('tournament_predictions', records, 'rank')

# --- entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tournament', action='store_true',
                        help='Also run tournament simulation')
    parser.add_argument('--sims', type=int, default=10_000,
                        help='Number of simulations (default: 10000)')
    args = parser.parse_args()

    print("Loading model and reference data...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    elo_dict, form_dict = load_refs()

    print("Parsing matches from local CSV...")
    df_live, df_upcoming = load_csv_fixtures()

    run_match_predictions(model, df_upcoming, elo_dict, form_dict)

    if args.tournament:
        run_tournament_simulation(model, df_live, df_upcoming, elo_dict, form_dict, n_sims=args.sims)

    print("\nDone.")