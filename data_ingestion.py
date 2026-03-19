import pandas as pd
from sqlalchemy import create_all_engine

def ingest_bts_data(db1b_path, t100_path):
    """
    Cleans messy BTS files and reconciles passenger demand with flight segments.
    """
    # Loading DB1B (True Demand)
    demand_df = pd.read_csv(db1b_path)
    
    # Cleaning SQL-style: Filtering out international connections
    # and normalizing airport codes for Dubuque (DBQ) [cite: 33, 70]
    clean_demand = demand_df[demand_df['Origin'] == 'DBQ']
    
    # Merging with T-100 (Actual Supply)
    supply_df = pd.read_csv(t100_path)
    
    # The 'Ghost Route' logic: Where demand exists but segments = 0
    # ... (pipeline continues)
    return merged_data