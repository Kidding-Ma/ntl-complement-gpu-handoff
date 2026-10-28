"""Kaggle CPU submission for Name That LLM.

This is the strongest locally verified CPU recipe in this workspace:
style-aware four-view TF-IDF + 25 handcrafted style statistics + LinearSVC.

Strict grouped 5-fold validation on the 2,699 official rows measured pooled
balanced accuracy 0.388936117. The vectorizers and StandardScaler are fit on
all official training rows here; the hidden Kaggle test is only transformed.

Paste this file into a Kaggle competition Notebook with Name That LLM data
attached. It writes the complete hidden-test submission to
/kaggle/working/submission.parquet. The local data/test.csv has only 3 examples
and must never be submitted as the competition answer.
"""
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")

EXPECTED_LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}
MIN_FULL_TEST_ROWS = 50
CANDIDATES = [
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
    Path("/kaggle/input/competition/name-that-llm"),
]
DATA_CANDIDATES = [p for p in CANDIDATES if (p / "train.csv").is_file() and (p / "test.csv").is_file()]
if len(DATA_CANDIDATES) != 1:
    raise RuntimeError(f"Expected exactly one complete competition data mount; found {DATA_CANDIDATES}")
DATA_DIR = DATA_CANDIDATES[0]

OUT = Path("/kaggle/working")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

required_train = {"id", "prompt_id", "text", "model"}
required_test = {"id", "text"}
assert required_train.issubset(train.columns), train.columns.tolist()
assert required_test.issubset(test.columns), test.columns.tolist()
assert train.id.is_unique and test.id.is_unique
assert len(test) >= MIN_FULL_TEST_ROWS, "Visible test is a smoke fixture, not a full competition test"
assert train[["id", "text", "model"]].notna().all().all()
assert test[["id", "text"]].notna().all().all()

classes = np.array(sorted(train.model.astype(str).unique()), dtype=object)
assert set(classes) == EXPECTED_LABELS, f"Unexpected competition labels: {set(classes) ^ EXPECTED_LABELS}"

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose this these those it "
    "its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to "
    "from with without for of by over under between through into about is are was were be been being have has "
    "had do does did can could will would shall should may might must not no nor all any some each every both "
    "either neither other another such only also just very too more most less least here there when where why "
    "how what whether then now still yet even much many few little".split()
)


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_text(text):
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def view_text(text, analyzer):
    if analyzer == "shape":
        return shape_text(text)
    if analyzer == "function":
        return function_text(text)
    return str(text)


def numeric_style(text):
    """Handcrafted style branch used in the best local CPU experiment."""
    text = str(text)
    length = max(len(text), 1)
    words = text.split()
    return [
        np.log1p(len(text)),
        np.log1p(len(words)),
        text.count("\n") / length * 100.0,
        text.count("\n\n"),
        *[text.count(ch) for ch in ["—", "–", ",", ".", ";", ":", '"', "'", "’", "“", "*", "#", "(", "-"]],
        sum(char.isupper() for char in text) / length,
        np.mean([len(word) for word in words]) if words else 0.0,
        len(re.findall(r"[.!?]", text)) / max(len(words), 1),
        *[text.count(ch) for ch in ["!", "?", "…", "..."]],
    ]


FEATURES = [
    ("char", 3, 5, 250000, 1.0),
    ("word", 1, 3, 180000, 0.7),
    ("shape", 2, 6, 60000, 0.5),
    ("function", 1, 3, 50000, 0.5),
]

train_texts = train.text.fillna("").astype(str).tolist()
test_texts = test.text.fillna("").astype(str).tolist()
train_parts = []
test_parts = []

for analyzer, lo, hi, max_features, weight in FEATURES:
    actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
    options = dict(
        analyzer=actual,
        ngram_range=(lo, hi),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=max_features,
        dtype=np.float64,
    )
    if actual == "word":
        options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vectorizer = TfidfVectorizer(**options)
    train_view = [view_text(text, analyzer) for text in train_texts]
    test_view = [view_text(text, analyzer) for text in test_texts]
    train_parts.append(vectorizer.fit_transform(train_view) * weight)
    test_parts.append(vectorizer.transform(test_view) * weight)
    print(f"完成 TF-IDF 分支: {analyzer}, 训练维度={train_parts[-1].shape[1]}", flush=True)

# Fit the scaler on official training rows only; the test rows are transform-only.
style_train = np.asarray([numeric_style(text) for text in train_texts], dtype=np.float64)
style_test = np.asarray([numeric_style(text) for text in test_texts], dtype=np.float64)
style_scaler = StandardScaler().fit(style_train)
NUMERIC_WEIGHT = 0.3
train_parts.append(csr_matrix(style_scaler.transform(style_train) * NUMERIC_WEIGHT))
test_parts.append(csr_matrix(style_scaler.transform(style_test) * NUMERIC_WEIGHT))
print(f"完成 numeric style 分支, 维度={style_train.shape[1]}", flush=True)

X_train = hstack(train_parts, format="csr")
X_test = hstack(test_parts, format="csr")
model = LinearSVC(C=0.5, random_state=42, max_iter=5000)
model.fit(X_train, train.model.astype(str))
pred = model.predict(X_test).astype(str)

submission = pd.DataFrame({"id": test.id.astype(str), "model": pred})[["id", "model"]]
assert len(submission) == len(test)
assert submission.id.is_unique
assert submission.id.tolist() == test.id.astype(str).tolist()
assert submission.model.notna().all()
assert submission.model.isin(classes).all()

output_path = OUT / "submission.parquet"
submission.to_parquet(output_path, index=False)
pd.testing.assert_frame_equal(submission, pd.read_parquet(output_path))

print("训练行数:", len(train))
print("测试行数:", len(test))
print("提交列:", submission.columns.tolist())
print("提交文件:", output_path)
print(submission["model"].value_counts().to_string())
print(submission.head(10).to_string(index=False))
print("提交文件已通过回读检查，可以点击 Submit to Competition。")
