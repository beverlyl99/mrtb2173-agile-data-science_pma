import pandas as pd
import sys

def validate_data(filepath):
    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)

    df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])
    issues = []

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n[FAIL] Missing values found:")
        for col, count in missing_cols.items():
            pct = round(count / len(df) * 100, 2)
            print(f"       {col}: {count} rows ({pct}%)")
        issues.append("missing_values")
    else:
        print("\n[PASS] No missing values")

    dups = df.duplicated().sum()
    if dups > 0:
        print(f"\n[FAIL] {dups} duplicate rows found ({round(dups/len(df)*100,2)}%)")
        issues.append("duplicates")
    else:
        print("\n[PASS] No duplicate rows")

    neg = (df["Quantity"] < 0).sum()
    if neg > 0:
        print(f"\n[FAIL] {neg} rows with negative Quantity")
        issues.append("negative_quantity")
    else:
        print("\n[PASS] No negative quantities")

    bad_price = (df["Price"] <= 0).sum()
    if bad_price > 0:
        print(f"\n[FAIL] {bad_price} rows with Price <= 0")
        issues.append("invalid_price")
    else:
        print("\n[PASS] All prices valid")

    print("\n" + "=" * 50)
    print(f"Total issues found: {len(issues)}")
    print("=" * 50)

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "online_retail_ii.csv"
    validate_data(filepath)
