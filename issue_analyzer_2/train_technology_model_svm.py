
import pandas as pd
import numpy as np
import os

from scipy.sparse import hstack

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("dataset_enriched.csv")

# One row per issue
df = df.drop_duplicates("issue_id").reset_index(drop=True)

print("Unique issues:", len(df))


# ============================================================
# 2. BUILD INPUT TEXT
# ============================================================

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

    language = (
        str(row["repository_language"])
        if pd.notna(row["repository_language"])
        else ""
    )

    return (
        "TITLE: " + title + " " +
        "TITLE: " + title + " " +
        "DESCRIPTION: " + description + " " +
        "LABELS: " + labels + " " +
        "LABELS: " + labels + " " +
        "LANGUAGE: " + language
    )


df["text"] = df.apply(build_text, axis=1)


# ============================================================
# 3. PREPARE TECHNOLOGY LABELS
# ============================================================

df["technology_list"] = df["issue_skills"].apply(
    lambda x: [
        skill.strip()
        for skill in str(x).split("|")
        if skill.strip()
    ]
)


# ============================================================
# 4. MULTI-LABEL ENCODING
# ============================================================

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(
    df["technology_list"]
)

print()
print("Number of technologies:", len(mlb.classes_))

print()
print("Technologies:")
print(list(mlb.classes_))


# ============================================================
# 5. ISSUE-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% train, 20% temporary
gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    gss1.split(
        df["text"],
        Y,
        groups=df["issue_id"]
    )
)

# Second: split temporary 50/50
# Gives approximately:
# 80% train
# 10% validation
# 10% test

gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

relative_val_idx, relative_test_idx = next(
    gss2.split(
        df.iloc[temp_idx]["text"],
        Y[temp_idx],
        groups=df.iloc[temp_idx]["issue_id"]
    )
)

val_idx = temp_idx[relative_val_idx]
test_idx = temp_idx[relative_test_idx]


X_train_text = df.iloc[train_idx]["text"]
X_val_text = df.iloc[val_idx]["text"]
X_test_text = df.iloc[test_idx]["text"]

Y_train = Y[train_idx]
Y_val = Y[val_idx]
Y_test = Y[test_idx]


print()
print("Training issues   :", len(train_idx))
print("Validation issues :", len(val_idx))
print("Testing issues    :", len(test_idx))


# ============================================================
# 6. WORD TF-IDF
# ============================================================

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=60000,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_val_word = word_vectorizer.transform(
    X_val_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)

print()
print("Word TF-IDF:", X_train_word.shape)


# ============================================================
# 7. CHARACTER TF-IDF
# ============================================================

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=40000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_val_char = char_vectorizer.transform(
    X_val_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    "Character TF-IDF:",
    X_train_char.shape
)


# ============================================================
# 8. COMBINE FEATURES
# ============================================================

X_train = hstack([
    X_train_word,
    X_train_char
])

X_val = hstack([
    X_val_word,
    X_val_char
])

X_test = hstack([
    X_test_word,
    X_test_char
])

print()
print("Combined train:", X_train.shape)
print("Combined val  :", X_val.shape)
print("Combined test :", X_test.shape)


# ============================================================
# 9. LINEAR SVM
# ============================================================

model = OneVsRestClassifier(
    LinearSVC(
        C=1.5,
        class_weight="balanced",
        max_iter=3000
    )
)

print()
print("Training Linear SVM...")

model.fit(
    X_train,
    Y_train
)


# ============================================================
# 10. VALIDATION SCORES
# ============================================================

val_scores = model.decision_function(
    X_val
)


# ============================================================
# 11. VALIDATION THRESHOLD SEARCH
# ============================================================

print()
print("=" * 60)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 60)

thresholds = np.arange(
    -1.0,
    1.01,
    0.10
)

best_threshold = 0.0
best_val_f1 = 0.0

for threshold in thresholds:

    Y_val_pred = (
        val_scores >= threshold
    ).astype(int)

    # Ensure at least one technology
    for i in range(len(Y_val_pred)):

        if Y_val_pred[i].sum() == 0:

            best_class = np.argmax(
                val_scores[i]
            )

            Y_val_pred[i, best_class] = 1

    score = f1_score(
        Y_val,
        Y_val_pred,
        average="micro",
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} -> "
        f"Validation Micro F1 {score:.4f}"
    )

    if score > best_val_f1:

        best_val_f1 = score
        best_threshold = threshold


print()
print(
    "Best validation threshold:",
    best_threshold
)

print(
    "Best validation Micro F1:",
    round(best_val_f1, 4)
)


# ============================================================
# 12. FINAL TEST PREDICTIONS
# ============================================================

test_scores = model.decision_function(
    X_test
)

Y_test_pred = np.zeros_like(
    test_scores,
    dtype=int
)


# ============================================================
# 13. TOP-K FILTERING
# ============================================================

TOP_K = 8

for i in range(len(test_scores)):

    valid_indices = np.where(
        test_scores[i] >= best_threshold
    )[0]

    if len(valid_indices) > TOP_K:

        top_indices = valid_indices[
            np.argsort(
                test_scores[i][valid_indices]
            )[-TOP_K:]
        ]

        Y_test_pred[
            i,
            top_indices
        ] = 1

    elif len(valid_indices) > 0:

        Y_test_pred[
            i,
            valid_indices
        ] = 1

    else:

        best_class = np.argmax(
            test_scores[i]
        )

        Y_test_pred[
            i,
            best_class
        ] = 1


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

micro_precision = precision_score(
    Y_test,
    Y_test_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    Y_test,
    Y_test_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    Y_test,
    Y_test_pred,
    average="micro",
    zero_division=0
)

macro_precision = precision_score(
    Y_test,
    Y_test_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    Y_test,
    Y_test_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    Y_test,
    Y_test_pred,
    average="macro",
    zero_division=0
)

hamming = hamming_loss(
    Y_test,
    Y_test_pred
)


print()
print("=" * 60)
print("TECHNOLOGY EXTRACTION - LINEAR SVM")
print("=" * 60)

print(
    f"Micro Precision : {micro_precision:.4f}"
)

print(
    f"Micro Recall    : {micro_recall:.4f}"
)

print(
    f"Micro F1        : {micro_f1:.4f}"
)

print()

print(
    f"Macro Precision : {macro_precision:.4f}"
)

print(
    f"Macro Recall    : {macro_recall:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print()

print(
    f"Hamming Loss    : {hamming:.4f}"
)


# ============================================================
# 15. PER-TECHNOLOGY F1
# ============================================================

per_tech_f1 = f1_score(
    Y_test,
    Y_test_pred,
    average=None,
    zero_division=0
)

print()
print("=" * 60)
print("PER-TECHNOLOGY F1")
print("=" * 60)

for technology, score in zip(
    mlb.classes_,
    per_tech_f1
):

    print(
        f"{technology}: {score:.4f}"
    )


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

results = df.iloc[test_idx][
    [
        "issue_id",
        "repository",
        "issue_title",
        "issue_description",
        "issue_skills"
    ]
].copy()


predicted_technologies = []

for row in Y_test_pred:

    technologies = [
        technology
        for technology, value in zip(
            mlb.classes_,
            row
        )
        if value == 1
    ]

    predicted_technologies.append(
        "|".join(technologies)
    )


results["predicted_technologies"] = (
    predicted_technologies
)


# ============================================================
# 17. SAVE CSV
# ============================================================

output_path = os.path.abspath(
    "technology_predictions_svm.csv"
)

results.to_csv(
    output_path,
    index=False
)


print()
print("=" * 60)
print("FILE SAVING")
print("=" * 60)

print("Saved successfully!")
print("File:", output_path)

print(
    "Rows:",
    len(results)
)

print(
    "File exists:",
    os.path.exists(output_path)
)

if os.path.exists(output_path):

    print(
        "File size:",
        os.path.getsize(output_path),
        "bytes"
    )

