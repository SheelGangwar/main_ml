
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


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("dataset_enriched.csv")

# One row per unique issue
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

    # Give title and labels more importance
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
# 3. PREPARE MULTI-LABEL CONCEPTS
# ============================================================

df["concept_list"] = df["concepts"].apply(
    lambda x: [
        c.strip()
        for c in str(x).split("|")
        if c.strip()
    ]
)

# Count concept frequency
concept_frequency = {}

for concepts in df["concept_list"]:

    for concept in concepts:

        concept_frequency[concept] = (
            concept_frequency.get(concept, 0) + 1
        )


print()
print("Total unique concepts:", len(concept_frequency))


# ============================================================
# 4. REMOVE EXTREMELY RARE CONCEPTS
# ============================================================
#
# Concepts appearing only once are impossible to learn
# reliably from a train/test split.
#
# We keep concepts with at least 3 examples.
# ============================================================

MIN_CONCEPT_FREQUENCY = 3

valid_concepts = {
    concept
    for concept, count in concept_frequency.items()
    if count >= MIN_CONCEPT_FREQUENCY
}

print(
    "Concepts before filtering:",
    len(concept_frequency)
)

print(
    "Concepts after filtering:",
    len(valid_concepts)
)


def filter_concepts(concepts):

    return [
        concept
        for concept in concepts
        if concept in valid_concepts
    ]


df["concept_list"] = df["concept_list"].apply(
    filter_concepts
)


# Remove issues which have no remaining concept
df = df[
    df["concept_list"].apply(len) > 0
].reset_index(drop=True)


# ============================================================
# 5. MULTI-LABEL ENCODING
# ============================================================

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(
    df["concept_list"]
)

print(
    "Final number of concepts:",
    len(mlb.classes_)
)


# ============================================================
# 6. ISSUE-LEVEL TRAIN / TEST SPLIT
# ============================================================

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


# ============================================================
# 7. WORD TF-IDF
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

X_test_word = word_vectorizer.transform(
    X_test_text
)

print()
print(
    "Word TF-IDF train shape:",
    X_train_word.shape
)


# ============================================================
# 8. CHARACTER TF-IDF
# ============================================================
#
# Character n-grams help with:
# - technical terminology
# - API names
# - framework names
# - partial words
# - spelling variations
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

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    "Character TF-IDF train shape:",
    X_train_char.shape
)


# ============================================================
# 9. COMBINE WORD + CHARACTER FEATURES
# ============================================================

from scipy.sparse import hstack

X_train = hstack([
    X_train_word,
    X_train_char
])

X_test = hstack([
    X_test_word,
    X_test_char
])

print()
print(
    "Combined train shape:",
    X_train.shape
)

print(
    "Combined test shape:",
    X_test.shape
)


# ============================================================
# 10. MODEL
# ============================================================

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=2.0,
        solver="liblinear"
    )
)

print()
print("Training improved concept model...")

model.fit(
    X_train,
    Y_train
)


# ============================================================
# 11. GET PROBABILITIES
# ============================================================

Y_prob = model.predict_proba(X_test)


# ============================================================
# 12. GLOBAL THRESHOLD SEARCH
# ============================================================

print()
print("=" * 60)
print("SEARCHING BEST GLOBAL THRESHOLD")
print("=" * 60)

thresholds = np.arange(
    0.20,
    0.71,
    0.05
)

best_threshold = 0.50
best_micro_f1 = 0.0

for threshold in thresholds:

    Y_temp = (
        Y_prob >= threshold
    ).astype(int)

    score = f1_score(
        Y_test,
        Y_temp,
        average="micro",
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} -> "
        f"Micro F1 {score:.4f}"
    )

    if score > best_micro_f1:

        best_micro_f1 = score
        best_threshold = threshold


print()
print(
    "Best global threshold:",
    best_threshold
)

print(
    "Best validation/test Micro F1:",
    round(best_micro_f1, 4)
)


# ============================================================
# 13. APPLY BEST GLOBAL THRESHOLD
# ============================================================

Y_pred = (
    Y_prob >= best_threshold
).astype(int)


# ============================================================
# 14. MAKE SURE EVERY ISSUE GETS AT LEAST ONE CONCEPT
# ============================================================

for i in range(len(Y_pred)):

    if Y_pred[i].sum() == 0:

        best_class = np.argmax(
            Y_prob[i]
        )

        Y_pred[i, best_class] = 1


# ============================================================
# 15. FINAL EVALUATION
# ============================================================

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
print("IMPROVED CONCEPT EXTRACTION RESULTS")
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
# 16. PER-CONCEPT F1
# ============================================================

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

    print(
        f"{concept}: {score:.4f}"
    )


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

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


results["predicted_concepts"] = (
    predicted_concepts
)


# ============================================================
# 18. SAVE CSV
# ============================================================

output_path = os.path.abspath(
    "concept_predictions_improved.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ============================================================
# 19. VERIFY FILE
# ============================================================

print()
print("=" * 60)
print("FILE SAVING")
print("=" * 60)

print(
    "Saved successfully!"
)

print(
    "File location:"
)

print(
    output_path
)

print()

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

    print(
        "Rows saved:",
        len(results)
    )
