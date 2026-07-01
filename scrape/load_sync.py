import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

# --- 1. SET UP THE ENGINES & CONFIG ---
db_filename = "worldcup_2026.db"
engine = create_engine(f"sqlite:///{db_filename}")
print(f"Connected to local SQLite database: '{db_filename}'")

username = os.environ.get("KAGGLE_USERNAME")
if not username:
    raise ValueError("Error: KAGGLE_USERNAME not found in .env")

supabase_url = os.environ.get("DATABASE_URL")
supabase_engine = None
if supabase_url:
    supabase_engine = create_engine(supabase_url)
    print("Cloud Supabase Engine Ready!")
else:
    print("Warning: No DATABASE_URL found in .env. Skipping Supabase mirroring.")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

KAGGLE_DATASETS = {
    "martj42/international-football-results-from-1872-to-2017": [
        "results.csv", "goalscorers.csv", "shootouts.csv"
    ],
    "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings": [
        "elo_ratings_wc2026.csv"
    ],
    "kulkarniparth09/fifa-world-cup-complete-dataset-19302026": [
        "wc_2026_fixtures.csv", "wc_2026_teams.csv", "wc_all_editions.csv", "wc_all_matches.csv", "wc_top_scorers.csv"
    ]
}

# --- 2. AUTOMATIC KAGGLE DOWNLOAD ---
def fetch_all_kaggle_data():
    print(f"\nAuthenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    for dataset_id in KAGGLE_DATASETS.keys():
        print(f"Downloading latest files from '{dataset_id}' into /{DATA_DIR}...")
        # Downloads and overwrites/updates the files in your local data folder
        api.dataset_download_files(dataset_id, path=DATA_DIR, unzip=True)
    print("All Kaggle datasets updated locally.")

# --- 3. HELPER FUNCTION TO BULK WRITE TABLES ---
def sync_table_to_databases(df, table_name, action="replace"):
    # Normalize column names to avoid database syntax bugs
    df.columns = [col.lower().replace(" ", "_").strip() for col in df.columns]
    
    # Write to Local SQLite
    print(f"  -> Syncing '{table_name}' to Local SQLite...")
    df.to_sql(table_name, con=engine, if_exists=action, index=False)
    
    # Write to Cloud Supabase Postgres
    if supabase_engine:
        print(f"  -> Syncing '{table_name}' to Cloud Supabase Postgres...")
        # method='multi' optimizes inserting chunks of data into Postgres
        df.to_sql(table_name, con=supabase_engine, if_exists=action, index=False, method='multi')

# --- 4. DATA PIPELINE LOOP ---
def run_pipeline():
    fetch_all_kaggle_data()
    datasets = {
        "elo_ratings_wc2026.csv": "elo_ratings_2026",
        "goalscorers.csv": "goalscorers",
        "shootouts.csv": "shootouts",
        "results.csv": "results",
        "wc_2026_fixtures.csv": "fixtures_2026",
        "wc_2026_teams.csv": "teams_2026",
        "wc_all_editions.csv": "wc_all_editions",
        "wc_all_matches.csv": "wc_all_matches",
        "wc_top_scorers.csv": "wc_top_scorers",
    }
    
    print("\n--- Syncing Standard Datasets ---")
    for csv_name, table_name in datasets.items():
        file_path = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            sync_table_to_databases(df, table_name)
        else:
            print(f"Warning: File {csv_name} not found in zip extract. Skipping.")

    # --- 5. THE RESULTS SPLIT EXTRACTION ---
    print("\n--- Parsing and Splitting ONLY 'FIFA World Cup 2026' ---")
    results_path = os.path.join(DATA_DIR, "results.csv")
    
    if not os.path.exists(results_path):
        raise FileNotFoundError("Critical Error: results.csv missing from the downloaded dataset.")
        
    columns = ['date', 'team1', 'team2', 'score1', 'score2', 'tournament', 'city', 'country', 'neutral']
    df_results = pd.read_csv(results_path, names=columns, header=None)
    
    # Keep ONLY the 2026 World Cup tournament matches
    # Option A: Filter by date range
    df_results = df_results[
        (df_results['date'] >= '2026-06-11') & 
        (df_results['date'] <= '2026-07-19') & 
        (df_results['tournament'].str.contains('World Cup', case=False, na=False))
    ].copy()

    # Sync the filtered main results file to your database
    sync_table_to_databases(df_results, "results")
    
    # Convert score columns to numbers so text flags like 'NA' become true NaNs
    df_results['score1'] = pd.to_numeric(df_results['score1'], errors='coerce')
    df_results['score2'] = pd.to_numeric(df_results['score2'], errors='coerce')
    
    # Dynamic split logic based on score presence
    df_live = df_results[df_results['score1'].notna()].copy()
    df_upcoming = df_results[df_results['score1'].isna()].copy()
    
    # Ensure baseline fallback structural columns exist for tracking
    if 'stage' not in df_live.columns:    df_live['stage'] = 'GROUP_STAGE'
    if 'stage' not in df_upcoming.columns: df_upcoming['stage'] = 'GROUP_STAGE'
    
    print(f"Detected {len(df_live)} completed 2026 WC matches and {len(df_upcoming)} upcoming 2026 WC fixtures.")
    
    # Synchronize split fixtures tables everywhere
    sync_table_to_databases(df_live, "fixtures_2026_live")
    sync_table_to_databases(df_upcoming, "fixtures_2026_upcoming")

if __name__ == "__main__":
    run_pipeline()