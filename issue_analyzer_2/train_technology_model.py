import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    classification_report
)


DATASET_PATH = "dataset_enriched.csv"


print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("Original shape:", df.shape)


# --------------------------------------------------
# 1. Keep one row per unique issue
# --------------------------------------------------

df = df.drop_duplicates(subset="issue_id").reset_index(drop=True)

print("Unique issues:", len(df))


# --------------------------------------------------
# 2. Clean required columns
# --------------------------------------------------

for col in [
    "issue_title",
    "issue_description",
    "issue_labels",
    "repository_language",
    "issue_skills"
]:
    df[col] = df[col].fillna("").astype(str)


# --------------------------------------------------
# 3. Create input text
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
# 4. Convert technology labels into lists
# --------------------------------------------------

df["technology_list"] = df["issue_skills"].apply(
    lambda x: [
        skill.strip().lower()
        for skill in x.split("|")
        if skill.strip()
    ]
)


# --------------------------------------------------
# 5. Encode multi-label target
# --------------------------------------------------

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(df["technology_list"])

print("\nTechnology labels:")
print(list(mlb.classes_))

print("\nNumber of technologies:", len(mlb.classes_))


# --------------------------------------------------
# 6. Issue-level train/test split
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
Y_test = Y[test_idx]


print("\nTraining issues:", len(train_idx))
print("Testing issues:", len(test_idx))


# --------------------------------------------------
# 7. TF-IDF
# --------------------------------------------------

print("\nTraining TF-IDF...")

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

print("Train TF-IDF shape:", X_train.shape)
print("Test TF-IDF shape:", X_test.shape)


# --------------------------------------------------
# 8. Multi-label Logistic Regression
# --------------------------------------------------

print("\nTraining model...")

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

model.fit(X_train, Y_train)

print("Training complete.")


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

Y_pred = model.predict(X_test)


# --------------------------------------------------
# 10. Evaluation
# --------------------------------------------------

print("\n" + "=" * 60)
print("TECHNOLOGY EXTRACTION RESULTS")
print("=" * 60)

micro_precision = precision_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0
)

macro_precision = precision_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0
)

hamming = hamming_loss(Y_test, Y_pred)


print(f"\nMicro Precision : {micro_precision:.4f}")
print(f"Micro Recall    : {micro_recall:.4f}")
print(f"Micro F1        : {micro_f1:.4f}")

print(f"\nMacro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")

print(f"\nHamming Loss    : {hamming:.4f}")


# --------------------------------------------------
# 11. Per-technology results
# --------------------------------------------------

print("\n" + "=" * 60)
print("PER-TECHNOLOGY RESULTS")
print("=" * 60)

print(
    classification_report(
        Y_test,
        Y_pred,
        target_names=mlb.classes_,
        zero_division=0
    )
)