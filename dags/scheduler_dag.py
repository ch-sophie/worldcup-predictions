import os
import subprocess
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

SCRIPTS_DIR = "/opt/airflow/scrape"

def run_worldcup_pipeline():
    """Triggers the unified loader script safely and logs execution failures."""
    script_path = os.path.join(SCRIPTS_DIR, "loader.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing deployment script at: {script_path}")
        
    print(f"Launching pipeline script: {script_path}")
    
    try:
        result = subprocess.run(
            ["python", "loader.py"],
            cwd=SCRIPTS_DIR,
            check=True,       # Forces Airflow to mark the task as FAILED if code crashes
            capture_output=True,
            text=True
        )
        print("Script output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Inner Script Crashed! Here is the error logs:")
        print(e.stderr) # This prints the actual error from loader.py into Airflow Logs
        raise e

# Default config
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'worldcup_sqlite_supabase_sync',
    default_args=default_args,
    description='Sync raw and cleaned tables across SQLite and Supabase every 6 hours',
    schedule_interval='0 */6 * * *', 
    catchup=False,                  
) as dag:

    execute_sync = PythonOperator(
        task_id='run_load_and_clean',
        python_callable=run_worldcup_pipeline,
    )