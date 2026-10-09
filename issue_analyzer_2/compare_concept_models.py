
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


DATA_PATH = "dataset_enriched.csv"
RANDOM_STATE = 42
TOP_K = 5


def build_text(row):
    return (
        f"{row['issue_title']} {row['issue_title']} "
        f"{row['issue_description']} "
        f"{row['issue_labels']} {row['issue_labels']} "
        f"{row['repository_language']}"
    )


def apply_top_k(scores, threshold, max_k=5):
    predictions = np.zeros_like(scores, dtype=int)

    for i in range(scores.shape[0]):
        valid = np.where(scores[i] >= threshold)[0]

        if len(valid) > max_k:
            order = np.argsort(scores[i][valid])[::-1]
            valid = valid[order[:max_k]]

        if len(valid) == 0:
            valid = [np.argmax(scores[i])]

        predictions[i, valid] = 1

    return predictions


def find_best_threshold(model, X_val, y_val):
    scores = model.decision_function(X_val)

    best_threshold = 0.0
    best_f1 = -1

    thresholds = np.arange(-1.0, 1.01, 0.05)

    for threshold in thresholds:
        pred = apply_top_k(
            scores,
            threshold,
            max_k=TOP_K
        )

        f1 = f1_score(
            y_val,
            pred,
            average="micro",
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["issue_id", "concepts"]
)

df = df.drop_duplicates(
    subset=["issue_id"],
    keep="first"
).reset_index(drop=True)


df["text"] = df.apply(
    build_text,
    axis=1
)


def parse_concepts(value):
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


df["concept_list"] = df["concepts"].apply(
    parse_concepts
)


df = df[
    df["concept_list"].apply(len) > 0
].reset_index(drop=True)


mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(
    df["concept_list"]
)


groups = df["issue_id"].values


gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    gss1.split(
        df,
        Y,
        groups=groups
    )
)


gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_relative, test_relative = next(
    gss2.split(
        df.iloc[temp_idx],
        Y[temp_idx],
        groups=groups[temp_idx]
    )
)


val_idx = temp_idx[val_relative]
test_idx = temp_idx[test_relative]


X_train_text = df.iloc[train_idx]["text"]
X_val_text = df.iloc[val_idx]["text"]
X_test_text = df.iloc[test_idx]["text"]

y_train = Y[train_idx]
y_val = Y[val_idx]
y_test = Y[test_idx]


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


X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_val_word = word_vectorizer.transform(
    X_val_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
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


X_train = hstack(
    [X_train_word, X_train_char]
).tocsr()

X_val = hstack(
    [X_val_word, X_val_char]
).tocsr()

X_test = hstack(
    [X_test_word, X_test_char]
).tocsr()


models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    ),

    "Linear SVM": OneVsRestClassifier(
        LinearSVC(
            C=1.5,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    ),

    "SGD Classifier": OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE
        )
    )
}


results = []
trained_models = {}
thresholds = {}


print("\n" + "=" * 70)
print("CONCEPT MODEL COMPARISON")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(
        X_train,
        y_train
    )

    threshold, val_micro_f1 = find_best_threshold(
        model,
        X_val,
        y_val
    )

    val_scores = model.decision_function(
        X_val
    )

    val_pred = apply_top_k(
        val_scores,
        threshold,
        max_k=TOP_K
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    trained_models[name] = model
    thresholds[name] = threshold

    results.append({
        "model": name,
        "threshold": threshold,
        "validation_micro_f1": val_micro_f1,
        "validation_macro_f1": val_macro_f1
    })

    print(
        f"Best Threshold : {threshold:.2f}"
    )

    print(
        f"Validation Micro F1 : {val_micro_f1:.4f}"
    )

    print(
        f"Validation Macro F1 : {val_macro_f1:.4f}"
    )


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "validation_micro_f1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL RANKING")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


best_model_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_model_name
]

best_threshold = thresholds[
    best_model_name
]


print("\n" + "=" * 70)
print("FINAL TEST - BEST MODEL")
print("=" * 70)

print(
    f"Selected Model : {best_model_name}"
)

print(
    f"Threshold      : {best_threshold:.2f}"
)


test_scores = best_model.decision_function(
    X_test
)

test_pred = apply_top_k(
    test_scores,
    best_threshold,
    max_k=TOP_K
)


test_micro_f1 = f1_score(
    y_test,
    test_pred,
    average="micro",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)


print(
    f"Test Micro F1 : {test_micro_f1:.4f}"
)

print(
    f"Test Macro F1 : {test_macro_f1:.4f}"
)


results_df.to_csv(
    "concept_model_comparison.csv",
    index=False
)


print("\nSaved:")
print("concept_model_comparison.csv")


