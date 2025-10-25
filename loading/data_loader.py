import os
import pandas as pd

def load_all_data(folder, limit=None):
    all_data = []
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    if limit:
        files = files[:limit]
    
    print(f"📂 Loading {len(files)} files...")
    
    for i, file in enumerate(files, 1):
        filepath = os.path.join(folder, file)
        df = pd.read_csv(filepath)
        df['ticker'] = file.replace('_minute.csv', '')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        all_data.append(df)

        if i % 10 == 0 or i == len(files):
            print(f"✅ Loaded {i}/{len(files)} files")

    return pd.concat(all_data, ignore_index=True)
