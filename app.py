import streamlit as st
import pandas as pd
import sqlite3
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="2026 World Cup ML",
    page_icon="🏆",
    layout="wide"
)

# --- 1. SET UP SQLITE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "worldcup_2026.db")

# --- 2. CACHED DATA FETCHING ---
@st.cache_data(ttl=120) 
def fetch_live_matches():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT * FROM fixtures_2026_live"
            df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            if 'team1' in df.columns:
                df['team1'] = df['team1'].astype(str).str.title()
            if 'team2' in df.columns:
                df['team2'] = df['team2'].astype(str).str.title()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error fetching live matches from SQLite: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600) 
def fetch_upcoming_matches():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT * FROM fixtures_2026_upcoming"
            df = pd.read_sql_query(query, conn)

        if not df.empty:
            if 'team1' in df.columns:
                df['team1'] = df['team1'].astype(str).str.title()
            if 'team2' in df.columns:
                df['team2'] = df['team2'].astype(str).str.title()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error fetching upcoming matches from SQLite: {e}")
        return pd.DataFrame()

# Load raw datasets
df_live_raw = fetch_live_matches()
df_upcoming_raw = fetch_upcoming_matches()

# --- 3. DYNAMIC SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filter Options")

# Collect unique countries across both datasets for the country filter
all_countries = set()
for df in [df_live_raw, df_upcoming_raw]:
    if not df.empty:
        if 'team1' in df.columns:
            all_countries.update(df['team1'].dropna().unique())
        if 'team2' in df.columns:
            all_countries.update(df['team2'].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    options=sorted(list(all_countries)),
    placeholder="All Countries"
)

# Date Filter (Date range slider based on dataset boundaries)
all_dates = []
for df in [df_live_raw, df_upcoming_raw]:
    if not df.empty and 'date' in df.columns:
        all_dates.extend(df['date'].dropna().tolist())

if all_dates:
    min_date = min(all_dates).date()
    max_date = max(all_dates).date()
    
    # If there's only 1 unique date or error boundaries, pad it
    if min_date == max_date:
        from datetime import timedelta
        max_date = min_date + timedelta(days=1)
        
    start_date, end_date = st.sidebar.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM DD, YYYY"
    )
else:
    start_date, end_date = None, None

# --- 4. APPLY FILTERS TO DATAFRAMES ---
def apply_filters(df):
    if df.empty:
        return df
    filtered_df = df.copy()
    
    # Country Filter logic (match if selected country is team1 OR team2)
    if selected_countries:
        filtered_df = filtered_df[
            filtered_df['team1'].isin(selected_countries) | 
            filtered_df['team2'].isin(selected_countries)
        ]
        
    # Date Filter logic
    if start_date and end_date and 'date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    return filtered_df

df_live = apply_filters(df_live_raw)
df_upcoming = apply_filters(df_upcoming_raw)

if not df_upcoming.empty and 'date' in df_upcoming.columns:
    df_upcoming['date'] = df_upcoming['date'].dt.date

# --- 5. MAIN INTERFACE ---
st.title("🏆 FIFA World Cup 2026 ML Predictor")
st.markdown("---")

tab_live, tab_upcoming, tab_predictions = st.tabs([
    "🔴 Past Matches", 
    "🗓️ Upcoming Matchups", 
    "🤖 Machine Learning Predictions"
])

# --- TAB 1: PAST MATCHES ---
with tab_live:
    st.subheader("⚡ PAST MATCHES")
    if df_live.empty:
        st.info("No games match the current filter selection.")
    else:
        for _, match in df_live.iterrows():
            with st.container(border=True):
                goals1 = int(match.get('score1', 0) if pd.notnull(match.get('score1')) else 0)
                goals2 = int(match.get('score2', 0) if pd.notnull(match.get('score2')) else 0)
                
                if goals1 > goals2:
                    color1 = "color: #28a745;"
                    color2 = "color: #dc3545;" 
                elif goals2 > goals1:
                    color1 = "color: #dc3545;"
                    color2 = "color: #28a745;"
                else:
                    color1 = "color: #6c757d;"  
                    color2 = "color: #6c757d;"

                col1, col2, col3, col4 = st.columns([3, 1, 1, 3])
                with col1:
                    st.markdown(f"### {match.get('team1', 'TBD')}")
                with col2:
                    st.markdown(f"<h2 style='{color1}'>{goals1}</h2>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<h2 style='{color2}'>{goals2}</h2>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"### {match.get('team2', 'TBD')}")
                
                # Format and display the past match date
                date_str = match['date'].strftime('%B %d, %Y') if 'date' in match and pd.notnull(match['date']) else "Date Unknown"
                st.caption(f"**Date:** {date_str} | Stage: {match.get('stage', 'Group Stage')}")

# --- TAB 2: UPCOMING MATCHES ---
with tab_upcoming:
    st.subheader("Schedule")
    if df_upcoming.empty:
        st.info("No upcoming scheduled games match the filter selections.")
    else:
        df_upcoming_sorted = df_upcoming.sort_values(by='date' if 'date' in df_upcoming.columns else df_upcoming.columns[0])
        st.dataframe(df_upcoming_sorted, use_container_width=True, hide_index=True)

# --- TAB 3: ML PREDICTIONS ---
with tab_predictions:
    st.subheader("🤖 Model-Generated Winner Probabilities")
    if df_upcoming.empty:
        st.info("No upcoming matches available for current filters.")
    else:
        for _, match in df_upcoming.head(5).iterrows():
            home_win_pct = int(match.get('ml_home_win_prob', 50) if pd.notnull(match.get('ml_home_win_prob')) else 50)
            away_win_pct = int(match.get('ml_away_win_prob', 50) if pd.notnull(match.get('ml_away_win_prob')) else 50)
            
            with st.container(border=True):
                date_str = match['date'].strftime('%B %d, %Y') if 'date' in match and pd.notnull(match['date']) else "Upcoming"
                st.write(f"**{match.get('stage', 'Tournament Stage')}** • {date_str}")
                
                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    st.metric(label=f"{match.get('team1')} Win Chance", value=f"{home_win_pct}%")
                with c2:
                    st.markdown("<h3 style='text-align: center; margin-top:15px;'>VS</h3>", unsafe_allow_html=True)
                with c3:
                    st.metric(label=f"{match.get('team2')} Win Chance", value=f"{away_win_pct}%")
                st.progress(home_win_pct)