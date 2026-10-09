import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


DATASET_PATH = "dataset_enriched.csv"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

df = df.drop_duplicates(subset="issue_id").reset_index(drop=True)

for col in [
    "issue_title",
    "issue_description",
    "issue_labels",
    "repository_language",
    "issue_skills"
]:
    df[col] = df[col].fillna("").astype(str)


# --------------------------------------------------
# 2. Build input text
# --------------------------------------------------

def build_text(row):
    return (
        "TITLE: " + row["issue_title"] + " "
        "DESCRIPTION: " + row["issue_description"] + " "
        "LABELS: " + row["issue_labels"] + " "
        "REPOSITORY_LANGUAGE: " + row["repository_language"]
    )


df["text"] = df.apply(build_text, axis=1)


# --------------------------------------------------
# 3. Prepare technology labels
# --------------------------------------------------

df["technology_list"] = df["issue_skills"].apply(
    lambda x: [
        skill.strip().lower()
        for skill in x.split("|")
        if skill.strip()
    ]
)

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(df["technology_list"])


# --------------------------------------------------
# 4. Same issue-level train/test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df["text"],
        Y,
        groups=df["issue_id"]
    )
)

X_train_text = df.iloc[train_idx]["text"]
X_test_text = df.iloc[test_idx]["text"]

Y_train = Y[train_idx]


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

print("Training model...")
model.fit(X_train, Y_train)
print("Training complete.")


# --------------------------------------------------
# 7. Predict
# --------------------------------------------------

Y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Show predictions
# --------------------------------------------------

test_df = df.iloc[test_idx].copy()

test_df["actual"] = [
    ", ".join(labels)
    for labels in mlb.inverse_transform(
        Y[test_idx]
    )
]

test_df["predicted"] = [
    ", ".join(labels)
    for labels in mlb.inverse_transform(
        Y_pred
    )
]


# --------------------------------------------------
# 9. Display 20 examples
# --------------------------------------------------

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

for i, (_, row) in enumerate(test_df.head(20).iterrows(), 1):

    print(f"\n{'-' * 80}")
    print(f"Example {i}")

    print("\nTITLE:")
    print(row["issue_title"])

    print("\nREPOSITORY:")
    print(row["repository"])

    print("\nACTUAL TECHNOLOGIES:")
    print(row["actual"])

    print("\nPREDICTED TECHNOLOGIES:")
    print(row["predicted"])


# --------------------------------------------------
# 10. Save predictions
# --------------------------------------------------

output = test_df[
    [
        "issue_id",
        "repository",
        "issue_title",
        "actual",
        "predicted"
    ]
]

output.to_csv(
    "technology_predictions.csv",
    index=False
)

print("\n")
print("Saved: technology_predictions.csv")