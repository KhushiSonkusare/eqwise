import os
import pandas as pd
from pathlib import Path

def load_all_data_chunked(input_folder, out_folder="tmp_parquets", limit=None, save_parquet=True):
    """
    Memory-safe data loader that reads multiple CSVs, processes them one by one,
    and saves each to Parquet format to avoid memory explosion.
    """
    os.makedirs(out_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if limit:
        files = files[:limit]

    print(f"📂 Found {len(files)} CSV files in '{input_folder}'")

    for i, file in enumerate(files, 1):
        filepath = os.path.join(input_folder, file)
        ticker = file.replace('_minute.csv', '')
        out_path = os.path.join(out_folder, f"{ticker}.parquet")

        if os.path.exists(out_path):
            print(f"⏩ Skipping {ticker} (already processed)")
            continue

        try:
            # Stream load with limited memory
            df_iter = pd.read_csv(filepath, chunksize=5_000_000)  # adjust chunk size to fit RAM
            for j, chunk in enumerate(df_iter, 1):
                chunk['ticker'] = ticker
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

                # Save to parquet incrementally
                mode = "append" if j > 1 else "overwrite"
                chunk.to_parquet(out_path, index=False, compression="snappy")

                print(f"✅ [{i}/{len(files)}] {ticker} chunk {j} saved ({len(chunk):,} rows)")

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    print("🎉 All files processed and saved to Parquet format")

def combine_parquets(folder="tmp_parquets", output_file="data_full.parquet"):
    """
    Combine all per-ticker Parquet files into one large Parquet file (still efficient).
    """
    pfiles = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    print(f"📁 Combining {len(pfiles)} parquet files into {output_file}")

    # Efficient concatenation
    combined = pd.concat((pd.read_parquet(f) for f in pfiles), ignore_index=True)
    combined.to_parquet(output_file, index=False, compression="snappy")

    # Optionally save as pickle
    combined.to_pickle(output_file.replace(".parquet", ".pkl"))
    print(f"✅ Combined dataset saved as {output_file}")
    print(f"✅ Rows: {len(combined):,}, Columns: {list(combined.columns)}")

    return combined


if __name__ == "__main__":
    # Step 1: Convert CSVs → individual Parquet files
    load_all_data_chunked("data", out_folder="tmp_parquets")

    # Step 2: Combine Parquet files into one final dataset
    data = combine_parquets("tmp_parquets", "data_full.parquet")

    # Summary
    print("✅ Unique stocks:", data['ticker'].nunique())
    print("✅ Date range:", data['date'].min(), "→", data['date'].max())
