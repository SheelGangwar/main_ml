
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
DATASET_PATH = "dataset_enriched.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Keep one record per unique issue
df = df.drop_duplicates(subset="issue_id").reset_index(drop=True)

print("Unique issues:", len(df))


# 2. Clean columns
text_columns = [
    "issue_title",
    "issue_description",
    "issue_labels",
    "repository_language",
    "difficulty"
]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Remove rows with missing difficulty
df = df[df["difficulty"] != ""]

# Normalize difficulty labels
df["difficulty"] = df["difficulty"].str.lower()

print("\nDifficulty distribution:")
print(df["difficulty"].value_counts())


# 3. Build input text
def build_text(row):
    return (
        "TITLE: " + row["issue_title"] + " "
        "DESCRIPTION: " + row["issue_description"] + " "
        "LABELS: " + row["issue_labels"] + " "
        "REPOSITORY_LANGUAGE: " + row["repository_language"]
    )

df["text"] = df.apply(build_text, axis=1)

X = df["text"]
y = df["difficulty"]


# 4. Train/test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining issues:", len(X_train_text))
print("Testing issues:", len(X_test_text))


# 5. TF-IDF
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


# 6. Train Logistic Regression
print("\nTraining difficulty model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training complete!")


# 7. Predict
y_pred = model.predict(X_test)


# 8. Evaluation
print("\n" + "=" * 60)
print("DIFFICULTY CLASSIFICATION RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f}")

print(f"Precision : {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

print(f"Recall    : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

print(f"Macro F1  : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")


# 9. Per-class results
print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        labels=["beginner", "intermediate", "advanced"],
        zero_division=0
    )
)


# 10. Confusion matrix
print("\nCONFUSION MATRIX")
print("Class order: beginner, intermediate, advanced")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["beginner", "intermediate", "advanced"]
    )
)


# 11. Save test predictions
results = pd.DataFrame({
    "actual_difficulty": y_test.values,
    "predicted_difficulty": y_pred
})

results.to_csv("difficulty_predictions.csv", index=False)

print("\nPredictions saved to difficulty_predictions.csv")