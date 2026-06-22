import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Establish connection to a local SQLite database file
db_filename = "world_cup_2026.db"
engine = create_engine(f"sqlite:///{db_filename}")
print(f"Connected to local database file: '{db_filename}' successfully!")

data_dir = "data"

# 2. Map of CSV files
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
        
        # Standardize column headers to lowercase and underscores
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # --- Team Name Standardization Dictionary ---
        # Forces common variations to match one master standard string
        name_map = {
            "usa": "united states",
            "us": "united states",
            "united states of america": "united states",
            "south korea": "korea republic",
            "republic of korea": "korea republic",
        }
        
        # Identify typical column names that contain team/country strings
        team_cols = ["team", "team_name", "team_1", "team_2", "team1", "team2", "country"]
        
        for col in team_cols:
            if col in df.columns:
                # 1. Convert text to lowercase, strip trailing/leading spaces, and convert to string
                df[col] = df[col].astype(str).str.lower().str.strip()
                # 2. Replace variations using dict mapping
                df[col] = df[col].replace(name_map)
        # --------------------------------------------
        
        print(f"Uploading to database table: '{table_name}' using strategy '{action}'...")
        df.to_sql(table_name, con=engine, if_exists=action, index=False)
        print(f"✅ '{table_name}' is live inside the .db file!\n")
    else:
        print(f"❌ Could not find {csv_name} in the '{data_dir}' folder.")