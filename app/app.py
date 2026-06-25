import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="2026 World Cup ML", page_icon="🏆", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "worldcup_2026.db"))

# --- data fetchers ---
@st.cache_data(ttl=120)
def fetch_live_matches():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM fixtures_2026_live", conn)
        if not df.empty:
            df['team1'] = df['team1'].str.title()
            df['team2'] = df['team2'].str.title()
            df['date']  = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_upcoming():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM fixtures_2026_upcoming", conn)
        if not df.empty:
            df['team1'] = df['team1'].str.title()
            df['team2'] = df['team2'].str.title()
            df['date']  = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()

@st.cache_data(ttl=120)
def fetch_predictions():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM predictions_2026", conn)
        if not df.empty:
            df['team1']      = df['team1'].str.title()
            df['team2']      = df['team2'].str.title()
            df['prediction'] = df['prediction'].str.title()
            df['date']       = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_tournament():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT rank, team, win_pct, final_pct, semifinal_pct "
                "FROM tournament_predictions ORDER BY rank", conn
            )
        return df
    except:
        return pd.DataFrame()

df_live_raw = fetch_live_matches()
df_up_raw   = fetch_upcoming()
df_preds    = fetch_predictions()
df_tourney  = fetch_tournament()

# --- sidebar filters ---
st.sidebar.header("🔍 Filter Options")

all_countries = set()
for df in [df_live_raw, df_up_raw]:
    if not df.empty:
        all_countries.update(df['team1'].dropna())
        all_countries.update(df['team2'].dropna())

selected = st.sidebar.multiselect("Select Countries:", sorted(all_countries), placeholder="All Countries")

all_dates = []
for df in [df_live_raw, df_up_raw]:
    if not df.empty and 'date' in df.columns:
        all_dates.extend(df['date'].dropna().tolist())

if all_dates:
    from datetime import timedelta
    mn, mx = min(all_dates).date(), max(all_dates).date()
    if mn == mx: mx = mn + timedelta(days=1)
    start_date, end_date = st.sidebar.slider("Date Range:", min_value=mn, max_value=mx,
                                              value=(mn, mx), format="MMM DD, YYYY")
else:
    start_date = end_date = None

def apply_filters(df):
    if df.empty: return df
    f = df.copy()
    if selected:
        f = f[f['team1'].isin(selected) | f['team2'].isin(selected)]
    if start_date and end_date and 'date' in f.columns:
        f = f[(f['date'].dt.date >= start_date) & (f['date'].dt.date <= end_date)]
    return f

df_live = apply_filters(df_live_raw)
df_up   = apply_filters(df_up_raw)
if not df_preds.empty and selected:
    df_preds = df_preds[df_preds['team1'].isin(selected) | df_preds['team2'].isin(selected)]
if not df_up.empty and 'date' in df_up.columns:
    df_up['date'] = df_up['date'].dt.date

# --- main UI ---
st.title("🏆 FIFA World Cup 2026 ML Predictor")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Past Matches", "🗓️ Upcoming", "🤖 Match Predictions", "🏆 Tournament Winner"
])

# TAB 1: Past matches
with tab1:
    st.subheader("Past Matches")
    if df_live.empty:
        st.info("No matches found for current filters.")
    else:
        for _, m in df_live.iterrows():
            g1 = int(m.get('score1', 0) or 0)
            g2 = int(m.get('score2', 0) or 0)
            c1 = "#28a745" if g1 > g2 else "#dc3545" if g1 < g2 else "#6c757d"
            c2 = "#28a745" if g2 > g1 else "#dc3545" if g2 < g1 else "#6c757d"
            with st.container(border=True):
                a, b, c, d = st.columns([3, 1, 1, 3])
                with a: st.markdown(f"### {m.get('team1','TBD')}")
                with b: st.markdown(f"<h2 style='color:{c1}'>{g1}</h2>", unsafe_allow_html=True)
                with c: st.markdown(f"<h2 style='color:{c2}'>{g2}</h2>", unsafe_allow_html=True)
                with d: st.markdown(f"### {m.get('team2','TBD')}")
                ds = m['date'].strftime('%B %d, %Y') if pd.notnull(m.get('date')) else "—"
                st.caption(f"**{ds}** | {m.get('stage','Group Stage')}")

# TAB 2: Upcoming
with tab2:
    st.subheader("Upcoming Schedule")
    if df_up.empty:
        st.info("No upcoming matches for current filters.")
    else:
        st.dataframe(df_up.sort_values('date') if 'date' in df_up.columns else df_up,
                     use_container_width=True, hide_index=True)

# TAB 3: Match predictions
with tab3:
    st.subheader("🤖 Match Win Probabilities")
    if df_preds.empty:
        st.info("Run `python 03_train_model.py` to generate predictions.")
    else:
        for _, m in df_preds.iterrows():
            p1  = float(m.get('prob_team1_win', 33))
            pd_ = float(m.get('prob_draw', 33))
            p2  = float(m.get('prob_team2_win', 33))
            pred = m.get('prediction', '—')
            with st.container(border=True):
                dv  = m.get('date')
                ds  = pd.to_datetime(dv).strftime('%b %d, %Y') if pd.notnull(dv) else "Upcoming"
                stg = str(m.get('stage', '')).replace('_', ' ').title()
                st.caption(f"{stg} • {ds}")
                c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 2, 3])
                with c1:
                    st.markdown(f"### {m.get('team1','?')}")
                    st.metric("Win", f"{p1:.1f}%")
                with c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(int(p1))
                with c3:
                    st.markdown(f"<h3 style='text-align:center;margin-top:10px'>VS</h3>"
                                f"<p style='text-align:center;color:#888'>Draw<br><b>{pd_:.1f}%</b></p>",
                                unsafe_allow_html=True)
                with c4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(int(p2))
                with c5:
                    st.markdown(f"### {m.get('team2','?')}")
                    st.metric("Win", f"{p2:.1f}%")
                st.markdown(f"<div style='text-align:center'>🤖 Predicted: <b>{pred}</b></div>",
                            unsafe_allow_html=True)

# TAB 4: Tournament winner
with tab4:
    st.subheader("🏆 Tournament Winner Probabilities")
    st.caption("Re-run `python simulate.py` to refresh after new results.")

    if df_tourney.empty:
        st.info("Run `python simulate.py` to generate tournament predictions.")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("#### Top 5 Favourites")
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for i, row in df_tourney.head(5).iterrows():
                st.metric(
                    label=f"{medals[i]} {row['team']}",
                    value=f"{row['win_pct']:.1f}%",
                    delta=f"Reach final: {row['final_pct']:.1f}%"
                )