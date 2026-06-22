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

# 2. Create a map of CSV files to the SQL tables
datasets = {
    "wc_2026_fixtures.csv": "fixtures_2026",
    "wc_2026_teams.csv": "teams_2026",
    "elo_ratings_wc2026.csv": "elo_ratings_2026",
    "former_names.csv": "former_names_2026",
    "goalscorers.csv": "goalscorers",
    "results.csv": "results",
    "shootouts.csv": "shootouts",
    "wc_all_editions.csv": "wc_all_editions",
    "wc_all_matches.csv": "wc_all_matches",
    "wc_top_scorers.csv": "wc_top_scorers",
}

# 3. Process and Load
for csv_name, table_name in datasets.items():
    file_path = os.path.join(data_dir, csv_name)
    
    if os.path.exists(file_path):
        print(f"Reading {csv_name}...")
        df = pd.read_csv(file_path)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        print(f"Uploading to database table: '{table_name}'...")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, method='multi')
        print(f"✅ '{table_name}' is live!\n")
    else:
        print(f"❌ Could not find {csv_name} in the '{data_dir}' folder.")

# --- Quick verification check --- 
with engine.connect() as connection:
    fixtures_df = pd.read_sql_query(text("SELECT * FROM fixtures_2026 LIMIT 5;"), connection)
    print("--- Fixtures Preview ---")
    print(fixtures_df)
    
    teams_df = pd.read_sql_query(text("SELECT * FROM teams_2026 LIMIT 5;"), connection)
    print("\n--- Teams Preview ---")
    print(teams_df)

    elo_df = pd.read_sql_query(text("SELECT * FROM elo_ratings_2026 LIMIT 5;"), connection)
    print("\n--- ELO Ratings Preview ---")
    print(elo_df)