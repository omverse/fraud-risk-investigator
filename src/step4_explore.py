import pandas as pd
import numpy as np
import time

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

def main():
    print("--- 1. Loading train_transaction.csv ---")
    t0 = time.time()
    train_trans = pd.read_csv("data/train_transaction.csv")
    print(f"Loaded train_transaction in {time.time() - t0:.2f}s")
    print(f"Shape: {train_trans.shape[0]} rows, {train_trans.shape[1]} columns")
    print(f"Column dtypes breakdown:\n{train_trans.dtypes.value_counts()}")

    print("\n--- 2. Calculate Fraud Rate ---")
    total_rows = len(train_trans)
    fraud_count = train_trans['isFraud'].sum()
    fraud_rate = train_trans['isFraud'].mean()
    print(f"Total rows: {total_rows:,}")
    print(f"Fraud count: {fraud_count:,}")
    print(f"Fraud rate overall: {fraud_rate:.5f} ({fraud_rate * 100:.3f}%)")

    print("\n--- 3. Checking Missing Values in train_transaction ---")
    missing_counts = train_trans.isnull().sum()
    missing_pct = missing_counts / total_rows * 100
    cols_with_missing = missing_counts[missing_counts > 0]
    print(f"Total columns with missing values: {len(cols_with_missing)} / {train_trans.shape[1]}")
    print(f"Columns with >50% missing values: {(missing_pct > 50).sum()}")
    print(f"Columns with >90% missing values: {(missing_pct > 90).sum()}")
    print("Top 10 columns by missing percentage:")
    print(pd.DataFrame({'missing_count': missing_counts, 'missing_pct': missing_pct}).sort_values('missing_pct', ascending=False).head(10))

    print("\n--- 4. Loading train_identity.csv and Left-Joining ---")
    t0 = time.time()
    train_id = pd.read_csv("data/train_identity.csv")
    print(f"Loaded train_identity in {time.time() - t0:.2f}s")
    print(f"train_identity shape: {train_id.shape[0]} rows, {train_id.shape[1]} columns")
    
    t0 = time.time()
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    print(f"Merged shape: {df.shape[0]} rows, {df.shape[1]} columns in {time.time() - t0:.2f}s")
    
    del train_trans, train_id
    import gc
    gc.collect()

    print("\n--- 5. Reduce Memory by Converting Number Columns to Smaller Types ---")
    df = reduce_mem_usage(df)
    
    # Save a small summary of step 4 results to verify
    summary = {
        "total_rows": total_rows,
        "fraud_count": int(fraud_count),
        "fraud_rate_overall": float(fraud_rate),
        "total_columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage().sum() / 1024**2)
    }
    print("\nStep 4 Summary:", summary)

if __name__ == "__main__":
    main()
