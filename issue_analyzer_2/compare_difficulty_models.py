
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score


DATA_PATH = "dataset_enriched.csv"
RANDOM_STATE = 42


df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["issue_id", "difficulty"]
)

df = df.drop_duplicates(
    subset=["issue_id"],
    keep="first"
).reset_index(drop=True)


def build_text(row):
    return (
        f"{row['issue_title']} "
        f"{row['issue_description']} "
        f"{row['issue_labels']} "
        f"{row['repository_language']}"
    )


df["text"] = df.apply(build_text, axis=1)


X_text = df["text"]
y = df["difficulty"]


X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


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


X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)
X_test_word = word_vectorizer.transform(X_test_text)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)
X_test_char = char_vectorizer.transform(X_test_text)


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
    "Logistic Regression":LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
), 

    "Linear SVM": LinearSVC(
        C=1.5,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    ),

    "SGD Classifier": SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE
    )
}


results = []
trained_models = {}


print("\n" + "=" * 70)
print("DIFFICULTY MODEL COMPARISON")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "model": name,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1
    })

    trained_models[name] = model

    print(
        f"Validation Accuracy : {val_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1 : {val_macro_f1:.4f}"
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_macro_f1",
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


print("\n" + "=" * 70)
print("FINAL TEST - BEST MODEL")
print("=" * 70)

print(
    f"Selected Model : {best_model_name}"
)


test_pred = best_model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
    zero_division=0
)


print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Macro F1 : {test_macro_f1:.4f}"
)


results_df.to_csv(
    "difficulty_model_comparison.csv",
    index=False
)


print("\nSaved:")
print("difficulty_model_comparison.csv")
