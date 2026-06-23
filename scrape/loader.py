import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Import the updated cleaning routine from your cleaner file
try:
    from cleaner import run_cleaning_pipeline
except ImportError:
    run_cleaning_pipeline = None

# --- 1. SET UP BOTH ENGINES ---
load_dotenv()

# Engine A: Local SQLite File
db_filename = "worldcup_2026.db"
engine = create_engine(f"sqlite:///{db_filename}")
print(f"Connected to local database file: '{db_filename}' successfully!")

# Engine B: Cloud Supabase Postgres
supabase_url = os.environ.get("DATABASE_URL")
supabase_engine = None

if supabase_url:
    supabase_engine = create_engine(supabase_url)
    print("☁️ Cloud Supabase Engine Ready!")
else:
    print("⚠️ Warning: No DATABASE_URL found. Skipping Supabase upload.")

# --- 2. THE PROCESSING LOOP (RAW TABLES) ---
data_dir = "data"
datasets = {
    "wc_2026_fixtures.csv": {"table": "fixtures_2026", "action": "replace"},
    "wc_2026_teams.csv": {"table": "teams_2026", "action": "replace"},
    "elo_ratings_wc2026.csv": {"table": "elo_ratings_2026", "action": "replace"},
    "former_names.csv": {"table": "former_names_2026", "action": "replace"},
    "goalscorers.csv": {"table": "goalscorers", "action": "replace"},
    "results.csv": {"table": "results", "action": "replace"},
    "shootouts.csv": {"table": "shootouts", "action": "replace"},
    "wc_all_editions.csv": {"table": "wc_all_editions", "action": "replace"},
    "wc_all_matches.csv": {"table": "wc_all_matches", "action": "replace"},
    "wc_top_scorers.csv": {"table": "wc_top_scorers", "action": "replace"},
}

for csv_name, config in datasets.items():
    file_path = os.path.join(data_dir, csv_name)
    table_name = config["table"]
    action = config["action"]
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # 🚀 Write to Database #1: Local SQLite
        print(f"Writing '{table_name}' to Local SQLite...")
        df.to_sql(table_name, con=engine, if_exists=action, index=False)
        
        # 🚀 Write to Database #2: Cloud Supabase
        if supabase_engine:
            print(f"Writing '{table_name}' to Cloud Supabase...")
            df.to_sql(table_name, con=supabase_engine, if_exists=action, index=False, method='multi')
            
        print(f"✅ '{table_name}' successfully updated everywhere!\n")

# --- 3. TRIGGER DUAL CLEANING PIPELINE ---
if run_cleaning_pipeline:
    print("=" * 60)
    print("All raw tables loaded! Launching dual cleaning pipelines...")
    print("=" * 60)
    
    #Clean Local SQLite
    print("\n[LOCAL] Cleaning local SQLite tables...")
    run_cleaning_pipeline(engine, db_type="SQLite")
    
    #Clean Cloud Supabase
    if supabase_engine:
        print("\n[CLOUD] Cleaning cloud Supabase tables...")
        run_cleaning_pipeline(supabase_engine, db_type="Supabase")
        
    print("\n🎉 Done! All raw and clean tables are matching locally and in Supabase.")
else:
    print("❌ Could not trigger cleaning: cleaner.py file is missing or formatted incorrectly.")