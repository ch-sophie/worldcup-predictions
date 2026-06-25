import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

local_engine = create_engine("sqlite:///worldcup_2026.db")
supabase_engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# --- 1. FETCH ---
def fetch_live_api_data():
    print("Fetching live 2026 World Cup data from football-data.org...")
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    headers = {
        "X-Auth-Token": FOOTBALL_DATA_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"Total matches returned: {data.get('resultSet', {}).get('count', 0)}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data: {e}")
        return None

# --- 2. CLEAN ---
def clean_api_data(json_data):
    if not json_data or "matches" not in json_data:
        print("No valid data received from the API.")
        return pd.DataFrame(), pd.DataFrame()

    print("Cleaning API data...")

    NAME_MAP = {
        "usa":                      "united states",
        "united states of america": "united states",
        "south korea":              "korea republic",
        "republic of korea":        "korea republic",
        "ir iran":                  "iran",
        "ivory coast":              "cote d'ivoire",
        "cape verde islands":       "cape verde",
        "trinidad & tobago":        "trinidad and tobago",
    }

    clean_matches = []
    for match in json_data["matches"]:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]

        if not home or not away:
            continue

        clean_matches.append({
            "match_id":   match["id"],
            "date":       match["utcDate"].split("T")[0],
            "stage":      match["stage"],
            "team1":      home.lower().strip(),
            "team2":      away.lower().strip(),
            "score1":     match["score"]["fullTime"]["home"],
            "score2":     match["score"]["fullTime"]["away"],
            "status":     match["status"],
            "played":     1 if match["status"] == "FINISHED" else 0
        })

    df = pd.DataFrame(clean_matches)
    if df.empty:
        return df, df

    df["team1"] = df["team1"].replace(NAME_MAP)
    df["team2"] = df["team2"].replace(NAME_MAP)

    df_played   = df[df["status"] == "FINISHED"].copy()
    df_upcoming = df[df["status"].isin(["SCHEDULED", "TIMED"])].copy()

    print(f"Played: {len(df_played)} | Upcoming: {len(df_upcoming)}")
    return df_played, df_upcoming

# --- 3. WRITE ---
def write_to_engines(df_played, df_upcoming):
    engines = {"SQLite": local_engine}
    if supabase_engine:
        engines["Supabase"] = supabase_engine

    for db_name, engine in engines.items():
        print(f"\nWriting to {db_name}...")

        if not df_played.empty:
            df_played[["match_id", "date", "stage", "team1", "team2", "score1", "score2", "played"]]\
                .to_sql("fixtures_2026_live", con=engine, if_exists="replace", index=False)
            print(f"fixtures_2026_live ({len(df_played)} matches)")

        if not df_upcoming.empty:
            df_upcoming[["match_id", "date", "stage", "team1", "team2"]]\
                .to_sql("fixtures_2026_upcoming", con=engine, if_exists="replace", index=False)
            print(f"fixtures_2026_upcoming ({len(df_upcoming)} matches)")

# --- MAIN ---
def main():
    raw_json = fetch_live_api_data()
    df_played, df_upcoming = clean_api_data(raw_json)

    if df_played.empty and df_upcoming.empty:
        print("No data to process.")
        return

    write_to_engines(df_played, df_upcoming)
    print("\nLive data pipeline complete!")

if __name__ == "__main__":
    main()