"""
Combine existing processed 15-minute data files
This script combines all the individual ticker files that were already processed
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
TMP_DIR = "tmp_15min_chunks"
OUT_PICKLE = "data_15min_with_indicators.pkl"
OUT_PARQUET = "data_15min_with_indicators.parquet"

def combine_processed_files():
    """Combine all processed ticker files into final dataset."""
    print("Combining existing processed files...")
    
    # Check if temp directory exists
    if not os.path.exists(TMP_DIR):
        print(f"ERROR: Directory {TMP_DIR} not found!")
        return
    
    # Find all processed files
    pfiles = sorted([os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith("_15m.parquet")])
    
    if not pfiles:
        print("ERROR: No processed files found!")
        return
    
    print(f"Found {len(pfiles)} processed files to combine")
    
    # Combine files in smaller chunks to avoid memory issues
    print("Combining files in chunks to avoid memory issues...")
    
    chunk_size = 50  # Process 50 files at a time
    all_parts = []
    
    for i in range(0, len(pfiles), chunk_size):
        chunk_files = pfiles[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(pfiles) + chunk_size - 1)//chunk_size} ({len(chunk_files)} files)")
        
        chunk_parts = []
        for j, p in enumerate(chunk_files, 1):
            try:
                part = pd.read_parquet(p)
                chunk_parts.append(part)
                if j % 10 == 0:
                    print(f"   Read {j}/{len(chunk_files)} files in chunk")
            except Exception as e:
                print(f"   ERROR reading {p}: {e}")
                continue
        
        if chunk_parts:
            # Combine this chunk
            chunk_combined = pd.concat(chunk_parts, ignore_index=True)
            all_parts.append(chunk_combined)
            print(f"   Chunk combined: {len(chunk_combined):,} rows")
            
            # Clear memory
            del chunk_parts, chunk_combined
    
    if not all_parts:
        print("ERROR: No data to combine!")
        return
    
    print("Combining all chunks into final dataset...")
    final = pd.concat(all_parts, ignore_index=True)
    
    print("Sorting by ticker and date...")
    final = final.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"Final shape: {final.shape}")
    print(f"Memory usage: {final.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # Final data type adjustments
    print("Final data type adjustments...")
    final['target'] = final['target'].astype("Int8")
    print("Target column converted to Int8")
    
    # Save outputs
    print(f"Saving final parquet: {OUT_PARQUET}")
    final.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print("Parquet file saved")
    
    print(f"Saving final pickle: {OUT_PICKLE}")
    final.to_pickle(OUT_PICKLE)
    print("Pickle file saved")
    
    print("Combination Complete!")
    print(f"Final dataset: {len(final):,} rows, {len(final.columns)} columns")
    print(f"Output files: {OUT_PARQUET}, {OUT_PICKLE}")
    print(f"Unique tickers: {final['ticker'].nunique()}")
    print(f"Date range: {final['date'].min()} to {final['date'].max()}")
    
    # Show column info
    print(f"\nColumns: {list(final.columns)}")
    
    # Clean up temp files (optional)
    response = input("\nDelete temporary files? (y/N): ")
    if response.lower() == 'y':
        for f in pfiles:
            try:
                os.remove(f)
            except:
                pass
        print("Temporary files cleaned up")

if __name__ == "__main__":
    combine_processed_files()
