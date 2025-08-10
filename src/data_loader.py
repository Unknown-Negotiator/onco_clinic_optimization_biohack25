import pandas as pd

def load_data(drug_paths, event_paths=None):
    drug_dfs = [pd.read_csv(p) for p in drug_paths]
    event_dfs = [pd.read_csv(p) for p in event_paths] if event_paths else None
    return drug_dfs, event_dfs
