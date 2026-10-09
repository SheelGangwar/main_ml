
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("dataset_enriched(1).csv")

# One row per unique issue
df = df.drop_duplicates("issue_id").reset_index(drop=True)

print("Unique issues:", len(df))


# ==========================================
# 2. BUILD INPUT TEXT
# ==========================================

def build_text(row):

    title = (
        str(row["issue_title"])
        if pd.notna(row["issue_title"])
        else ""
    )

    description = (
        str(row["issue_description"])
        if pd.notna(row["issue_description"])
        else ""
    )

    labels = (
        str(row["issue_labels"])
        if pd.notna(row["issue_labels"])
        else ""
    )

    repo_language = (
        str(row["repository_language"])
        if pd.notna(row["repository_language"])
        else ""
    )

    return (
        "TITLE: " + title +
        " DESCRIPTION: " + description +
        " LABELS: " + labels +
        " LANGUAGE: " + repo_language
    )


df["text"] = df.apply(build_text, axis=1)


# ==========================================
# 3. PREPARE CONCEPT LABELS
# ==========================================

df["concept_list"] = df["concepts"].apply(
    lambda x: [
        concept.strip()
        for concept in str(x).split("|")
        if concept.strip()
    ]
)

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(df["concept_list"])

print("Number of concepts:", len(mlb.classes_))

print()
print("Concepts:")
for concept in mlb.classes_:
    print("-", concept)


# ==========================================
# 4. ISSUE-LEVEL TRAIN / TEST SPLIT
# ==========================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        df["text"],
        Y,
        groups=df["issue_id"]
    )
)

X_train_text = df.iloc[train_idx]["text"]
X_test_text = df.iloc[test_idx]["text"]

Y_train = Y[train_idx]
Y_test = Y[test_idx]

print()
print("Training issues:", len(train_idx))
print("Testing issues:", len(test_idx))


# ==========================================
# 5. TF-IDF
# ==========================================

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

print()
print("TF-IDF train shape:", X_train.shape)
print("TF-IDF test shape:", X_test.shape)


# ==========================================
# 6. MULTI-LABEL CLASSIFIER
# ==========================================

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

print()
print("Training concept model...")

model.fit(X_train, Y_train)


# ==========================================
# 7. PREDICTIONS
# ==========================================

Y_pred = model.predict(X_test)


# ==========================================
# 8. EVALUATION
# ==========================================

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

hamming = hamming_loss(
    Y_test,
    Y_pred
)


print()
print("=" * 60)
print("CONCEPT EXTRACTION RESULTS")
print("=" * 60)

print(f"Micro Precision : {micro_precision:.4f}")
print(f"Micro Recall    : {micro_recall:.4f}")
print(f"Micro F1        : {micro_f1:.4f}")

print()

print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")

print()

print(f"Hamming Loss    : {hamming:.4f}")


# ==========================================
# 9. PER-CONCEPT F1
# ==========================================

per_concept_f1 = f1_score(
    Y_test,
    Y_pred,
    average=None,
    zero_division=0
)

print()
print("=" * 60)
print("PER-CONCEPT F1")
print("=" * 60)

for concept, score in zip(
    mlb.classes_,
    per_concept_f1
):
    print(f"{concept}: {score:.4f}")


# ==========================================
# 10. CREATE TEST PREDICTIONS
# ==========================================

results = df.iloc[test_idx][
    [
        "issue_id",
        "repository",
        "issue_title",
        "issue_description",
        "concepts"
    ]
].copy()


predicted_concepts = []

for row in Y_pred:

    concepts = [
        concept
        for concept, value in zip(
            mlb.classes_,
            row
        )
        if value == 1
    ]

    predicted_concepts.append(
        "|".join(concepts)
    )


results["predicted_concepts"] = predicted_concepts


# ==========================================
# 11. SAVE CSV
# ==========================================

output_path = os.path.abspath(
    "concept_predictions.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ==========================================
# 12. VERIFY FILE
# ==========================================

print()
print("=" * 60)
print("FILE SAVING")
print("=" * 60)

print("Saved successfully!")
print("File location:")
print(output_path)

print()
print("File exists:", os.path.exists(output_path))

if os.path.exists(output_path):
    print(
        "File size:",
        os.path.getsize(output_path),
        "bytes"
    )

    print(
        "Rows saved:",
        len(results)
    )

