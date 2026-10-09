
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATASET = "services/skill_extraction/repo_skill_dataset.csv"
MODEL_FILE = "services/skill_extraction/repo_skill_classifier.joblib"

TEXT_COLUMNS = [
    "skill",
    "domain",
    "language_evidence",
    "dependency_evidence",
    "topic_evidence",
    "readme_evidence",
    "semantic_evidence",
    "evidence"
]


def create_text(data):
    return data[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)


def main():

    print("=" * 60)
    print("Repository Skill Classification - Model Training")
    print("=" * 60)

    df = pd.read_csv(DATASET)

    print("\nDataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    train_df = df[df["split"] == "train"].copy()
    validation_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("\nTrain:", len(train_df))
    print("Validation:", len(validation_df))
    print("Test:", len(test_df))

    X_train = create_text(train_df)
    y_train = train_df["label"]

    X_validation = create_text(validation_df)
    y_validation = validation_df["label"]

    X_test = create_text(test_df)
    y_test = test_df["label"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=20000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                C=1.5
            )
        )
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    validation_predictions = model.predict(X_validation)

    print(
        "Accuracy:",
        accuracy_score(y_validation, validation_predictions)
    )

    print(
        "Macro F1:",
        f1_score(
            y_validation,
            validation_predictions,
            average="macro"
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    test_predictions = model.predict(X_test)

    print(
        "Accuracy:",
        accuracy_score(y_test, test_predictions)
    )

    print(
        "Macro F1:",
        f1_score(
            y_test,
            test_predictions,
            average="macro"
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=["negative", "positive", "uncertain"]
        )
    )

    print("\nTraining final model...")

    train_validation_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True
    )

    X_train_final = create_text(train_validation_df)
    y_train_final = train_validation_df["label"]

    model.fit(X_train_final, y_train_final)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully!")
    print("File:", MODEL_FILE)


if __name__ == "__main__":
    main()

