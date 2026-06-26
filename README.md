# worldcup-predictions
Build an end-to-end system to predict football match outcomes for the 2026 WC

Local sqlite db
- scrape loader.py (with cleaner) -> live_scrape.py (live from free api limitations)
- model/predict.py
- model/simulate.py
- streamlit run app/app.py

Supabase
- scrape loader (with cleaner) -> live_scrape (supabse db)
- predict_supabase
- (simulate_supabase)
- streamlit run app_supabase
