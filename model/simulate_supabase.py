import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")

# --- Supabase Initialization ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Missing Supabase credentials!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

HOSTS = {'united states', 'mexico', 'canada'}
FEATURE_COLS = [
    'elo_diff',
    'team1_last5_pts', 'team2_last5_pts', 'pts_diff',
    'team1_last5_gd',  'team2_last5_gd',  'gd_diff',
    'team1_is_host',   'team2_is_host',
]

# --- reference data ---
def load_refs():
    # Fetch elo data from Supabase
    elo_res = supabase.table("elo_latest").select("country, elo_latest").execute()
    elo = pd.DataFrame(elo_res.data)
    elo['country'] = elo['country'].str.lower().str.strip()
    elo_dict = dict(zip(elo['country'], elo['elo_latest']))

    # Fetch form data from Supabase
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

# --- pre-compute prob cache ---
def build_prob_cache(all_teams, model, elo_dict, form_dict):
    pairs  = [(t1, t2) for t1 in all_teams for t2 in all_teams if t1 != t2]
    rows   = [build_features(t1, t2, elo_dict, form_dict) for t1, t2 in pairs]
    X      = pd.DataFrame(rows)[FEATURE_COLS]
    probs  = model.predict_proba(X)
    cmap   = {c: i for i, c in enumerate(model.classes_)}

    cache_gs = {}
    cache_ko = {}

    for (t1, t2), p in zip(pairs, probs):
        ph = p[cmap.get(2, 0)]
        pd_ = p[cmap.get(1, 0)]
        pa = p[cmap.get(0, 0)]
        cache_gs[(t1, t2)] = (ph, pd_, pa)
        total = ph + pa
        p1_ko = (ph + pd_ * ph / total) if total > 0 else 0.5
        cache_ko[(t1, t2)] = p1_ko

    return cache_gs, cache_ko

# --- group stage helpers ---
def load_group_data():
    # Fetch cleaner teams setup
    teams_res = supabase.table("teams_2026_clean").select("team, group").execute()
    df_teams = pd.DataFrame(teams_res.data)
    df_teams['team'] = df_teams['team'].str.lower().str.strip()

    # Fetch live matches
    live_res = supabase.table("fixtures_2026_live").select("team1, team2, score1, score2").eq("stage", "GROUP_STAGE").execute()
    df_live = pd.DataFrame(live_res.data)
    if not df_live.empty:
        df_live['team1'] = df_live['team1'].str.lower().str.strip()
        df_live['team2'] = df_live['team2'].str.lower().str.strip()
    else:
        df_live = pd.DataFrame(columns=['team1', 'team2', 'score1', 'score2'])

    # Fetch upcoming matches
    up_res = supabase.table("fixtures_2026_upcoming").select("team1, team2").eq("stage", "GROUP_STAGE").execute()
    df_up = pd.DataFrame(up_res.data)
    if not df_up.empty:
        df_up['team1'] = df_up['team1'].str.lower().str.strip()
        df_up['team2'] = df_up['team2'].str.lower().str.strip()
    else:
        df_up = pd.DataFrame(columns=['team1', 'team2'])

    groups = df_teams.groupby('group')['team'].apply(list).to_dict()
    return groups, df_live, df_up

def points_from_score(s1, s2):
    if s1 > s2: return 3, 0
    if s1 < s2: return 0, 3
    return 1, 1

def compute_base_standings(df_live, groups):
    pts = defaultdict(int)
    gd  = defaultdict(int)
    gf  = defaultdict(int)
    for _, r in df_live.iterrows():
        t1, t2 = r['team1'], r['team2']
        if pd.isnull(r['score1']) or pd.isnull(r['score2']):
            continue
        s1, s2 = int(r['score1']), int(r['score2'])
        p1, p2 = points_from_score(s1, s2)
        pts[t1] += p1; pts[t2] += p2
        gd[t1]  += s1-s2; gd[t2]  += s2-s1
        gf[t1]  += s1;    gf[t2]  += s2
    return pts, gd, gf

# --- vectorised simulation ---
def run_simulations(n_sims: int = 10_000):
    print("Loading model and reference data from Supabase...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found")
    model = joblib.load(MODEL_PATH)
    elo_dict, form_dict = load_refs()

    print("Loading group stage data...")
    groups, df_live, df_up = load_group_data()

    all_teams = [t for teams in groups.values() for t in teams]
    print(f"Pre-computing match probabilities for {len(all_teams)} teams...")
    cache_gs, cache_ko = build_prob_cache(all_teams, model, elo_dict, form_dict)

    base_pts, base_gd, base_gf = compute_base_standings(df_live, groups)

    upcoming_pairs = list(zip(df_up['team1'], df_up['team2']))
    print(f"Group stage: {len(df_live)} played, {len(upcoming_pairs)} remaining")
    print(f"\nRunning {n_sims:,} simulations...\n")

    champion_counts  = defaultdict(int)
    finalist_counts  = defaultdict(int)
    semifinal_counts = defaultdict(int)

    n_upcoming = len(upcoming_pairs)
    rands_gs = np.random.random((n_sims, n_upcoming))

    for sim in range(n_sims):
        pts = defaultdict(int, base_pts)
        gd  = defaultdict(int, base_gd)
        gf  = defaultdict(int, base_gf)

        for i, (t1, t2) in enumerate(upcoming_pairs):
            ph, pd_, pa = cache_gs.get((t1, t2), (0.4, 0.25, 0.35))
            r = rands_gs[sim, i]
            if r < ph:
                pts[t1] += 3; gd[t1] += 1; gf[t1] += 1
            elif r < ph + pd_:
                pts[t1] += 1; pts[t2] += 1
            else:
                pts[t2] += 3; gd[t2] += 1; gf[t2] += 1

        standings = {}
        third_place = []
        for grp, teams in groups.items():
            sorted_t = sorted(
                teams,
                key=lambda t: (pts[t], gd[t], gf[t], np.random.random()),
                reverse=True
            )
            standings[grp] = sorted_t
            if len(sorted_t) >= 3:
                t3 = sorted_t[2]
                third_place.append((pts[t3], gd[t3], gf[t3], t3))

        third_place.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best8 = [x[3] for x in third_place[:8]]

        w = {g: standings[g][0] for g in standings}
        r = {g: standings[g][1] for g in standings}
        r32 = [
            (w['A'], r['B']), (w['B'], r['A']),
            (w['C'], r['D']), (w['D'], r['C']),
            (w['E'], r['F']), (w['F'], r['E']),
            (w['G'], r['H']), (w['H'], r['G']),
            (w['I'], r['J']), (w['J'], r['I']),
            (w['K'], r['L']), (w['L'], r['K']),
            (best8[0], best8[1]), (best8[2], best8[3]),
            (best8[4], best8[5]), (best8[6], best8[7]),
        ]

        def ko_round(matchups):
            winners = []
            for t1, t2 in matchups:
                p1 = cache_ko.get((t1, t2), 0.5)
                winners.append(t1 if np.random.random() < p1 else t2)
            return winners

        r32w = ko_round(r32)
        r16w = ko_round([(r32w[i], r32w[i+1]) for i in range(0, 16, 2)])
        qfw  = ko_round([(r16w[i], r16w[i+1]) for i in range(0, 8, 2)])
        sfw  = ko_round([(qfw[i],  qfw[i+1])  for i in range(0, 4, 2)])

        for t in sfw:
            semifinal_counts[t] += 1

        final_winner = ko_round([(sfw[0], sfw[1])])[0]
        finalist_counts[sfw[0]] += 1
        finalist_counts[sfw[1]] += 1
        champion_counts[final_winner] += 1

        if (sim + 1) % 2000 == 0:
            print(f"  {sim+1:,} / {n_sims:,} done...")

    # --- build results ---
    rows = []
    for team in all_teams:
        rows.append({
            'team':          team.title(),
            'win_pct':       round(champion_counts[team]  / n_sims * 100, 2),
            'final_pct':     round(finalist_counts[team]  / n_sims * 100, 2),
            'semifinal_pct': round(semifinal_counts[team] / n_sims * 100, 2),
            'win_count':     champion_counts[team],
        })

    df_out = (pd.DataFrame(rows)
              .sort_values('win_pct', ascending=False)
              .reset_index(drop=True))
    
    # Standardize rank indexing starting from 1
    df_out.index += 1
    df_out = df_out.reset_index().rename(columns={'index': 'rank'})

    # --- Upload to Supabase ---
    print("\nUploading structural records to Supabase 'tournament_predictions'...")
    records = df_out.to_dict(orient="records")
    
    try:
        # Clear old predictions to behave exactly like if_exists='replace'
        supabase.table("tournament_predictions").delete().neq("rank", -1).execute()
        # Insert fresh simulation metrics
        supabase.table("tournament_predictions").insert(records).execute()
        print("Successfully synced simulation matrix to Supabase!")
    except Exception as e:
        print(f"Error updating Supabase database: {e}")

    # --- Local Display Matrix ---
    print(f"\n{'Rank':<5} {'Team':<22} {'🏆 Win%':>8} {'Final%':>8} {'Semi%':>8}")
    print("─" * 58)
    for i, row in df_out.head(16).iterrows():
        rank_val = row['rank']
        medal = "🥇" if rank_val == 1 else "🥈" if rank_val == 2 else "🥉" if rank_val == 3 else "  "
        print(f"{rank_val:<5} {medal} {row['team']:<20} "
              f"{row['win_pct']:>7.1f}%  "
              f"{row['final_pct']:>7.1f}%  "
              f"{row['semifinal_pct']:>7.1f}%")

    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sims', type=int, default=10_000)
    args = parser.parse_args()
    run_simulations(args.sims)