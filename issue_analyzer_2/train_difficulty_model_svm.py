import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,precision_score

)
from scipy.sparse import hstack


# =========================
# 1. LOAD DATA
# =========================

DATA_PATH = "dataset_enriched.csv"

df = pd.read_csv(DATA_PATH)

# One row per unique issue
df = df.drop_duplicates(subset="issue_id").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("\nDifficulty distribution:")
print(df["difficulty"].value_counts())


# =========================
# 2. PREPARE TEXT
# =========================

def clean_value(x):
    if pd.isna(x):
        return ""
    return str(x)


df["text"] = (
    "TITLE " + df["issue_title"].apply(clean_value) + " " +
    "TITLE " + df["issue_title"].apply(clean_value) + " " +
    "DESCRIPTION " + df["issue_description"].apply(clean_value) + " " +
    "LABELS " + df["issue_labels"].apply(clean_value) + " " +
    "LABELS " + df["issue_labels"].apply(clean_value) + " " +
    "REPO_LANGUAGE " + df["repository_language"].apply(clean_value)
)

X = df["text"]
y = df["difficulty"]


# =========================
# 3. TRAIN / VALIDATION / TEST
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplits:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# =========================
# 4. WORD TF-IDF
# =========================

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=60000,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)
X_test_word = word_vectorizer.transform(X_test)


# =========================
# 5. CHARACTER TF-IDF
# =========================

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=40000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)
X_test_char = char_vectorizer.transform(X_test)


# =========================
# 6. COMBINE FEATURES
# =========================

X_train_final = hstack([X_train_word, X_train_char])
X_val_final = hstack([X_val_word, X_val_char])
X_test_final = hstack([X_test_word, X_test_char])

print("\nFeature shapes:")
print("Train:", X_train_final.shape)
print("Validation:", X_val_final.shape)
print("Test:", X_test_final.shape)


# =========================
# 7. TRAIN LINEAR SVM
# =========================

model = LinearSVC(
    C=1.5,
    class_weight="balanced",
    max_iter=3000
)

print("\nTraining Difficulty SVM...")

model.fit(X_train_final, y_train)


# =========================
# 8. VALIDATION
# =========================

val_pred = model.predict(X_val_final)

val_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("Accuracy:", accuracy_score(y_val, val_pred))
print("Macro F1:", val_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        val_pred,
        labels=["beginner", "intermediate", "advanced"],
        zero_division=0
    )
)


# =========================
# 9. FINAL TEST
# =========================

test_pred = model.predict(X_test_final)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("Accuracy:", accuracy_score(y_test, test_pred))

print(
    "f1 score:",
    f1_score(y_test, test_pred, average="macro")
)
print(
    "Macro Precision:",
    precision_score(y_test, test_pred, average="macro", zero_division=0)
)
print("Macro F1:", f1_score(y_test, test_pred, average="macro"))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=["beginner", "intermediate", "advanced"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred,
        labels=["beginner", "intermediate", "advanced"]
    )
)


# =========================
# 10. SAVE PREDICTIONS
# =========================

results = df.loc[y_test.index, [
    "issue_id",
    "repository",
    "issue_title",
    "difficulty"
]].copy()

results["predicted_difficulty"] = test_pred

results.to_csv(
    "difficulty_predictions_svm.csv",
    index=False
)

print("\nSaved:")
print("difficulty_predictions_svm.csv")