# Football match predictions (FIFA World Cup 2026) 

- Repository: `worldcup-prediction`
- Type of Challenge: `Consolidation`
- Duration: `5 days`
- Where: `Becode (June 2026)`

## Learning Objectives
Build an end-to-end system that predicts football match outcomes using scraping, scheduling, machine learning, and data visualization.

At the end of this challenge:
- Scrape and process data from football websites
- Train a machine learning model on historical match data
- Create a Streamlit app for live data visualization and predictions
- Manage the entire data pipeline, scheduling for automation

### Dataset
- **[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)** (Kaggle) — every international match result since 1872: date, teams, score, tournament type, city/country, neutral venue flag. Good base for feature engineering (recent form, head-to-head, home advantage).
- **[FIFA World Cup Complete Dataset: 1930–2026](https://www.kaggle.com/datasets/kulkarniparth09/fifa-world-cup-complete-dataset-19302026)** (Kaggle) — match-level results for every World Cup edition, plus a full 2026 fixtures file (group, stage, venue, kick-off times, FIFA rankings, confederation) and a 2026 qualified-teams file (group, confederation, FIFA rank, coach, best-ever result).
- **[2026 FIFA World Cup — Historical Elo Ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)** (Kaggle) — Elo ratings per team with weekly/daily snapshots through the tournament, useful as a ready-made strength feature instead of computing Elo from scratch.
- **[FIFA World Cup Dataset](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset)** (Kaggle) — team-level, beginner-friendly, with historical features calculated only from past results before each edition (already split to avoid leakage), squad age/market value where available.
- **Football API** (from Football API.org) - live scores for 2026 World Cup (free tier limitations).

### Features
- **Model Training**:
  - Train a machine learning model on historical match data to predict the outcome of future World Cup matches.

- **Streamlit Dashboard**:
  - Display past World Cup matches and their scores
  - Display upcoming World Cup matches and predicted outcomes using the machine learning model
  - Display tournament winner by simulation (top 5 predicted winners)

- **Scraper**:
  - Build a scraper to fetch recent/live match data and results as the tournament progresses.

- **Automation**:
  - Automate scraping using a scheduling tool to update the data periodically.

- **Database**:
  - Connect to a local SQLite DB and Cloud Supabase.

#### To do:
- `resolve automation with live scrape blocked (only works manually)` 
- `works with kaggle with delay`
- `add knockout tab`