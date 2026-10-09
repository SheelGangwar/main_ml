import pandas as pd

df = pd.read_csv("dataset_enriched.csv")

print("Dataset shape:", df.shape)
print()

print("Concepts missing:", df["concepts"].isna().sum())
print("Concepts blank:", (df["concepts"].fillna("").str.strip() == "").sum())
print()

print("Sample concepts:")
for i, value in enumerate(df["concepts"].dropna().head(20)):
    print(f"{i+1}. {value}")

print()
print("Unique concept values:", df["concepts"].nunique())