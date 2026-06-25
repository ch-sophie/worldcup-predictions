import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# --- 1. SETUP ---
engine = create_engine("sqlite:///worldcup_2026.db")
MODEL_PATH = "model.pkl"
HOSTS = ['united states', 'mexico', 'canada']

# --- 2. LOAD REFERENCE DATA ---
def load_elo_dict():
    """Latest Elo rating per team from elo_latest table."""
    df = pd.read_sql_query("SELECT country, elo_latest FROM elo_latest", engine)
    df['country'] = df['country'].str.lower().str.strip()
    return dict(zip(df['country'], df['elo_latest']))

def load_form_dict():
    """Recent form stats per team from team_form table."""
    df = pd.read_sql_query("SELECT * FROM team_form", engine)
    df['team'] = df['team'].str.lower().str.strip()
    return df.set_index('team').to_dict(orient='index')

# --- 3. FEATURE ENGINEERING ---
def build_features(team1: str, team2: str, elo_dict: dict, form_dict: dict) -> dict:
    """
    Build feature dict for a single match (team1 vs team2).
    All names should be lowercase/stripped before calling.
    """
    elo1 = elo_dict.get(team1, 1500)
    elo2 = elo_dict.get(team2, 1500)

    form1 = form_dict.get(team1, {})
    form2 = form_dict.get(team2, {})

    return {
        # Strength
        'elo_diff': elo1 - elo2,
        # Form (last 5 matches)
        'team1_last5_pts':  form1.get('last5_pts', 7.5),   # fallback = median
        'team2_last5_pts':  form2.get('last5_pts', 7.5),
        'pts_diff':         form1.get('last5_pts', 7.5) - form2.get('last5_pts', 7.5),
        'team1_last5_gd':   form1.get('last5_gd', 0),
        'team2_last5_gd':   form2.get('last5_gd', 0),
        'gd_diff':          form1.get('last5_gd', 0) - form2.get('last5_gd', 0),

        # Host advantage
        'team1_is_host': int(team1 in HOSTS),
        'team2_is_host': int(team2 in HOSTS),
    }

FEATURE_COLS = [
    'elo_diff',
    'team1_last5_pts', 'team2_last5_pts', 'pts_diff',
    'team1_last5_gd',  'team2_last5_gd',  'gd_diff',
    'team1_is_host',   'team2_is_host',
]

# --- 4. BUILD TRAINING SET FROM HISTORICAL DATA ---
def build_training_data(elo_dict: dict, form_dict: dict) -> tuple:
    """
    Uses wc_matches_clean (WC history) + results_clean (all internationals)
    Returns X (DataFrame) and y (array: 0=away_win, 1=draw, 2=home_win)
    """
    label_map = {'home_win': 2, 'draw': 1, 'away_win': 0}

    # WC matches — weight more
    df_wc = pd.read_sql_query(
        "SELECT team1, team2, result FROM wc_matches_clean WHERE result IS NOT NULL",
        engine
    )
    df_wc['weight'] = 3.0   # WC matches count 3x

    # All international results — broader signal
    df_all = pd.read_sql_query(
        """
        SELECT home_team AS team1, away_team AS team2, result
        FROM results_clean
        WHERE result IS NOT NULL
          AND date >= '2000-01-01'   -- recent enough to be relevant
        """,
        engine
    )
    df_all['weight'] = 1.0

    df = pd.concat([df_wc, df_all], ignore_index=True)
    df['team1'] = df['team1'].str.lower().str.strip()
    df['team2'] = df['team2'].str.lower().str.strip()
    df['y'] = df['result'].map(label_map)
    df = df.dropna(subset=['y'])

    rows = []
    for _, row in df.iterrows():
        feats = build_features(row['team1'], row['team2'], elo_dict, form_dict)
        feats['y'] = int(row['y'])
        feats['weight'] = row['weight']
        rows.append(feats)

    df_features = pd.DataFrame(rows)
    X = df_features[FEATURE_COLS]
    y = df_features['y'].values
    weights = df_features['weight'].values

    print(f"Training set: {len(X)} matches  |  "
          f"home_win={( y==2).sum()}  draw={(y==1).sum()}  away_win={(y==0).sum()}")
    return X, y, weights

# --- 5. TRAIN AND EVALUATE ---
def train_model():
    print("Loading reference data...")
    elo_dict  = load_elo_dict()
    form_dict = load_form_dict()

    print("Building training data from historical matches...")
    X, y, weights = build_training_data(elo_dict, form_dict)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',   # handles draw imbalance
    )

    # Quick cross-validation sanity check
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Final fit on all data with weights
    model.fit(X, y, sample_weight=weights)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

# --- 6. PREDICT UPCOMING FIXTURES ---
def run_predictions():
    # Load or train model
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    else:
        print("No saved model found — training now...")
        model = train_model()

    elo_dict  = load_elo_dict()
    form_dict = load_form_dict()

    # Load upcoming fixtures
    df_upcoming = pd.read_sql_query(
        "SELECT * FROM fixtures_2026_upcoming", engine
    )

    if df_upcoming.empty:
        print("No upcoming fixtures found.")
        return

    print(f"\nGenerating predictions for {len(df_upcoming)} matches...\n")

    results = []
    for _, row in df_upcoming.iterrows():
        t1 = row['team1'].lower().strip()
        t2 = row['team2'].lower().strip()

        feats = build_features(t1, t2, elo_dict, form_dict)
        X_pred = pd.DataFrame([feats])[FEATURE_COLS]

        probs = model.predict_proba(X_pred)[0]

        # Map class indices back to labels
        # Classes order from model: 0=away_win, 1=draw, 2=home_win
        class_order = {c: i for i, c in enumerate(model.classes_)}
        prob_home = probs[class_order.get(2, 0)]
        prob_draw = probs[class_order.get(1, 0)]
        prob_away = probs[class_order.get(0, 0)]

        results.append({
            'match_id':       row['match_id'],
            'date':           row['date'],
            'stage':          row['stage'],
            'team1':          row['team1'],
            'team2':          row['team2'],
            'prob_team1_win': round(prob_home * 100, 1),
            'prob_draw':      round(prob_draw * 100, 1),
            'prob_team2_win': round(prob_away * 100, 1),
            'prediction':     (
                row['team1'] if prob_home >= prob_draw and prob_home >= prob_away
                else row['team2'] if prob_away >= prob_draw
                else 'Draw'
            ),
        })

    df_pred = pd.DataFrame(results)

    # Print summary
    print(f"{'Date':<12} {'Team 1':<20} {'Team 2':<20} {'T1 Win':>7} {'Draw':>7} {'T2 Win':>7}  Prediction")
    print("-" * 85)
    for _, r in df_pred.iterrows():
        print(
            f"{r['date']:<12} {r['team1']:<20} {r['team2']:<20} "
            f"{r['prob_team1_win']:>6.1f}% {r['prob_draw']:>6.1f}% {r['prob_team2_win']:>6.1f}%  "
            f"→ {r['prediction']}"
        )

    # Save to DB
    df_pred.to_sql("predictions_2026", con=engine, if_exists="replace", index=False)
    print(f"\nPredictions saved to 'predictions_2026' table.")

    return df_pred

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        train_model()
    else:
        run_predictions()