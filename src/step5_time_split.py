import pandas as pd
import numpy as np
import time

def reduce_mem_usage(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
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
    return df

def main():
    print("Loading data...")
    train_trans = pd.read_csv("data/train_transaction.csv")
    train_id = pd.read_csv("data/train_identity.csv")
    df = pd.merge(train_trans, train_id, on="TransactionID", how="left")
    del train_trans, train_id
    
    print("Downcasting memory...")
    df = reduce_mem_usage(df)
    
    # Check TransactionID and TransactionDT dtypes
    dtypes_check = df[['TransactionID', 'TransactionDT']].dtypes
    print("\n--- Check: TransactionID and TransactionDT dtypes ---")
    print(dtypes_check)
    for col in ['TransactionID', 'TransactionDT']:
        assert df[col].dtype in [np.int32, np.int64], f"{col} was downcasted too far: {df[col].dtype}"
    print("Check passed: Both are int32 or larger, ensuring no overflow!")

    print("\n--- Step 5: Time-based split ---")
    # 1. Sort by TransactionDT (stable sort)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    train_rows, train_fraud_rate = len(train_df), train_df['isFraud'].mean()
    val_rows, val_fraud_rate = len(val_df), val_df['isFraud'].mean()
    test_rows, test_fraud_rate = len(test_df), test_df['isFraud'].mean()

    train_min_t, train_max_t = train_df['TransactionDT'].min(), train_df['TransactionDT'].max()
    val_min_t, val_max_t = val_df['TransactionDT'].min(), val_df['TransactionDT'].max()
    test_min_t, test_max_t = test_df['TransactionDT'].min(), test_df['TransactionDT'].max()

    print(f"Train set:      {train_rows:,} rows ({train_rows/n_total*100:.1f}%) | Fraud rate: {train_fraud_rate:.5f} ({train_fraud_rate*100:.3f}%) | Min TransactionDT: {train_min_t:,} | Max TransactionDT: {train_max_t:,}")
    print(f"Validation set: {val_rows:,} rows ({val_rows/n_total*100:.1f}%) | Fraud rate: {val_fraud_rate:.5f} ({val_fraud_rate*100:.3f}%) | Min TransactionDT: {val_min_t:,} | Max TransactionDT: {val_max_t:,}")
    print(f"Test set:       {test_rows:,} rows ({test_rows/n_total*100:.1f}%) | Fraud rate: {test_fraud_rate:.5f} ({test_fraud_rate*100:.3f}%) | Min TransactionDT: {test_min_t:,} | Max TransactionDT: {test_max_t:,}")

    # Check temporal ordering
    print("\nTemporal boundary verification:")
    print(f"Train Max ({train_max_t:,}) <= Val Min ({val_min_t:,}): {train_max_t <= val_min_t}")
    print(f"Val Max ({val_max_t:,}) <= Test Min ({test_min_t:,}): {val_max_t <= test_min_t}")

if __name__ == "__main__":
    main()
