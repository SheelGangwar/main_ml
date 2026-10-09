import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


# ============================================================
# CONFIG
# ============================================================

DATASET = "dataset_enriched.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# COMMON TEXT PREPARATION
# ============================================================

def prepare_text(df):

    title = df["issue_title"].fillna("").astype(str)
    description = df["issue_description"].fillna("").astype(str)
    labels = df["issue_labels"].fillna("").astype(str)
    language = df["repository_language"].fillna("").astype(str)

    text = (
        "TITLE " + title + " "
        "TITLE " + title + " "
        "DESCRIPTION " + description + " "
        "LABELS " + labels + " "
        "LABELS " + labels + " "
        "LANGUAGE " + language
    )

    return text


# ============================================================
# ISSUE-LEVEL SPLIT
# ============================================================

def issue_split(df):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["issue_id"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=RANDOM_STATE
    )

    train_idx2, val_idx = next(
        splitter2.split(
            train_df,
            groups=train_df["issue_id"]
        )
    )

    final_train = train_df.iloc[train_idx2].copy()
    val_df = train_df.iloc[val_idx].copy()

    return final_train, val_df, test_df


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_vectorizers(train_text):

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=60000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=40000,
        sublinear_tf=True
    )

    word_vectorizer.fit(train_text)
    char_vectorizer.fit(train_text)

    return word_vectorizer, char_vectorizer


def transform_text(text, word_vectorizer, char_vectorizer):

    word_features = word_vectorizer.transform(text)
    char_features = char_vectorizer.transform(text)

    return hstack(
        [word_features, char_features],
        format="csr"
    )


# ============================================================
# TECHNOLOGY MODEL
# ============================================================

def train_technology_model(df):

    print("\n" + "=" * 70)
    print("TRAINING TECHNOLOGY MODEL")
    print("=" * 70)

    df = df.drop_duplicates("issue_id").reset_index(drop=True)

    df["text"] = prepare_text(df)

    df["technology_list"] = (
        df["issue_skills"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: [
                v.strip()
                for v in x.split("|")
                if v.strip()
            ]
        )
    )

    mlb = MultiLabelBinarizer()

    y = mlb.fit_transform(df["technology_list"])

    train_df, val_df, test_df = issue_split(df)

    train_ids = set(train_df["issue_id"])
    val_ids = set(val_df["issue_id"])

    train_mask = df["issue_id"].isin(train_ids)
    val_mask = df["issue_id"].isin(val_ids)
    test_mask = ~(train_mask | val_mask)

    train_text = df.loc[train_mask, "text"]

    word_vectorizer, char_vectorizer = build_vectorizers(
        train_text
    )

    X_train = transform_text(
        train_text,
        word_vectorizer,
        char_vectorizer
    )

    X_val = transform_text(
        df.loc[val_mask, "text"],
        word_vectorizer,
        char_vectorizer
    )

    X_test = transform_text(
        df.loc[test_mask, "text"],
        word_vectorizer,
        char_vectorizer
    )

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    model = OneVsRestClassifier(
        LinearSVC(
            C=1.5,
            class_weight="balanced",
            max_iter=3000
        )
    )

    print("Fitting Linear SVM...")

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    val_scores = model.decision_function(X_val)

    best_threshold = 0.0
    best_f1 = -1

    for threshold in np.arange(-1.0, 1.01, 0.10):

        pred = (val_scores >= threshold).astype(int)

        for i in range(len(pred)):
            if pred[i].sum() == 0:
                pred[i, np.argmax(val_scores[i])] = 1

        score = f1_score(
            y_val,
            pred,
            average="micro"
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    print("Best threshold:", best_threshold)
    print("Validation Micro F1:", best_f1)

    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "label_binarizer": mlb,
        "threshold": best_threshold,
        "top_k": 8,
        "model_type": "LinearSVC",
        "task": "technology_extraction"
    }

    path = os.path.join(
        MODEL_DIR,
        "technology_pipeline.pkl"
    )

    joblib.dump(artifact, path)

    print("Saved:", path)

    return artifact


# ============================================================
# DIFFICULTY MODEL
# ============================================================

def train_difficulty_model(df):

    print("\n" + "=" * 70)
    print("TRAINING DIFFICULTY MODEL")
    print("=" * 70)

    df = df.drop_duplicates("issue_id").reset_index(drop=True)

    df["text"] = prepare_text(df)

    labels = df["difficulty"].astype(str)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        stratify=train_df["difficulty"],
        random_state=RANDOM_STATE
    )

    train_text = train_df["text"]

    word_vectorizer, char_vectorizer = build_vectorizers(
        train_text
    )

    X_train = transform_text(
        train_df["text"],
        word_vectorizer,
        char_vectorizer
    )

    X_val = transform_text(
        val_df["text"],
        word_vectorizer,
        char_vectorizer
    )

    X_test = transform_text(
        temp_df["text"],
        word_vectorizer,
        char_vectorizer
    )

    y_train = train_df["difficulty"]
    y_val = val_df["difficulty"]
    y_test = temp_df["difficulty"]

    model = SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    )

    print("Fitting SGD Classifier...")

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    print(
        "Validation Macro F1:",
        f1_score(
            y_val,
            val_pred,
            average="macro"
        )
    )

    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "classes": model.classes_,
        "model_type": "SGDClassifier",
        "task": "difficulty_classification"
    }

    path = os.path.join(
        MODEL_DIR,
        "difficulty_pipeline.pkl"
    )

    joblib.dump(artifact, path)

    print("Saved:", path)

    return artifact


# ============================================================
# CONCEPT MODEL
# ============================================================

def train_concept_model(df):

    print("\n" + "=" * 70)
    print("TRAINING CONCEPT MODEL")
    print("=" * 70)

    df = df.drop_duplicates("issue_id").reset_index(drop=True)

    df["text"] = prepare_text(df)

    df["concept_list"] = (
        df["concepts"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: [
                v.strip()
                for v in x.split("|")
                if v.strip()
            ]
        )
    )

    mlb = MultiLabelBinarizer()

    y = mlb.fit_transform(df["concept_list"])

    train_df, val_df, test_df = issue_split(df)

    train_ids = set(train_df["issue_id"])
    val_ids = set(val_df["issue_id"])

    train_mask = df["issue_id"].isin(train_ids)
    val_mask = df["issue_id"].isin(val_ids)
    test_mask = ~(train_mask | val_mask)

    word_vectorizer, char_vectorizer = build_vectorizers(
        df.loc[train_mask, "text"]
    )

    X_train = transform_text(
        df.loc[train_mask, "text"],
        word_vectorizer,
        char_vectorizer
    )

    X_val = transform_text(
        df.loc[val_mask, "text"],
        word_vectorizer,
        char_vectorizer
    )

    X_test = transform_text(
        df.loc[test_mask, "text"],
        word_vectorizer,
        char_vectorizer
    )

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    model = OneVsRestClassifier(
        LinearSVC(
            C=1.5,
            class_weight="balanced",
            max_iter=3000
        )
    )

    print("Fitting Linear SVM...")

    model.fit(X_train, y_train)

    scores = model.decision_function(X_val)

    best_threshold = -0.30

    best_f1 = -1

    for threshold in np.arange(-1.0, 1.01, 0.10):

        pred = (scores >= threshold).astype(int)

        for i in range(len(pred)):

            positive = np.where(pred[i] == 1)[0]

            if len(positive) == 0:

                top_indices = np.argsort(
                    scores[i]
                )[-5:]

                pred[i, top_indices] = 1

        score = f1_score(
            y_val,
            pred,
            average="micro"
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(threshold)

    print("Selected threshold:", best_threshold)
    print("Validation Micro F1:", best_f1)

    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "label_binarizer": mlb,
        "threshold": best_threshold,
        "top_k": 5,
        "model_type": "LinearSVC",
        "task": "concept_extraction"
    }

    path = os.path.join(
        MODEL_DIR,
        "concept_pipeline.pkl"
    )

    joblib.dump(artifact, path)

    print("Saved:", path)

    return artifact


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET)

    print("Dataset:", df.shape)

    required = [
        "issue_id",
        "issue_title",
        "issue_description",
        "issue_labels",
        "repository_language",
        "issue_skills",
        "difficulty",
        "concepts"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: " + str(missing)
        )

    train_technology_model(df)

    train_difficulty_model(df)

    train_concept_model(df)

    print("\n" + "=" * 70)
    print("ALL MODEL ARTIFACTS SAVED")
    print("=" * 70)

    print("""
models/
    technology_pipeline.pkl
    difficulty_pipeline.pkl
    concept_pipeline.pkl
""")
