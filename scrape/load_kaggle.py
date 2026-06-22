import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

# 1. Establish connection to supabase
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ Error: Could not find DATABASE_URL in the .env file.")
    exit()

engine = create_engine(DATABASE_URL)
print("Connected to database securely!")

data_dir = "data"

# 2. Map of CSV files, their SQL tables, and how to handle existing data
# Using 'replace' for full-snapshot datasets ensures to never get duplicate rows
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

# 3. Process and Load
for csv_name, config in datasets.items():
    file_path = os.path.join(data_dir, csv_name)
    table_name = config["table"]
    action = config["action"]
    
    if os.path.exists(file_path):
        print(f"Reading {csv_name}...")
        df = pd.read_csv(file_path)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        print(f"Uploading to database table: '{table_name}' using strategy '{action}'...")
        df.to_sql(table_name, con=engine, if_exists=action, index=False, method='multi')
        print(f"✅ '{table_name}' is live!\n")
    else:
        print(f"❌ Could not find {csv_name} in the '{data_dir}' folder.")