# run_system.py - optional convenience script
import subprocess
import sys
import time

def run_ingestor():
    return subprocess.Popen([sys.executable, "ingestion.py"])

def run_streamlit():
    # streamlit CLI
    return subprocess.Popen(["streamlit", "run", "dashboard_streamlit.py"])

if __name__ == "__main__":
    p1 = run_ingestor()
    time.sleep(1)
    p2 = run_streamlit()
    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        print("Stopping services...")
        p1.terminate()
        p2.terminate()
