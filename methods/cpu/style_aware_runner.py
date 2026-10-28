"""Prompt-grouped OOF runner for the style-aware CPU recipe.

This script writes only private artifacts. It requires a private data directory
and an explicit output directory; no competition text is committed by default.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

SEED = 42
FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose this these those it its "
    "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from "
    "with without for of by over under between through into about is are was were be been being have has had do "
    "does did can could will would shall should may might must not no nor all any some each every both either "
    "neither other another such only just very too more most less least here there when where why how what whether "
    "then now still yet even much many few little".split()
)
FEATURES = [
    ("char", 3, 5, False, True, 250000, 1.0),
    ("word", 1, 3, False, True, 180000, 0.7),
    ("shape", 2, 6, False, True, 60000, 0.5),
    ("function", 1, 3, False, True, 50000, 0.5),
]


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_text(text):
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def transform_text(text, analyzer):
    return shape_text(text) if analyzer == "shape" else function_text(text) if analyzer == "function" else str(text)


def build_features(fit_texts, other_texts):
    fit_parts, other_parts = [], [[] for _ in other_texts]
    for analyzer, lo, hi, lower, sublinear, maximum, weight in FEATURES:
        actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
        options = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower,
                      sublinear_tf=sublinear, min_df=2, max_features=maximum, dtype=np.float64)
        if actual == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        fit_view = [transform_text(x, analyzer) for x in fit_texts]
        fit_parts.append(vectorizer.fit_transform(fit_view) * weight)
        for j, texts in enumerate(other_texts):
            other_parts[j].append(vectorizer.transform([transform_text(x, analyzer) for x in texts]) * weight)
    return hstack(fit_parts, format="csr"), [hstack(x, format="csr") for x in other_parts]


def provider(label):
    for prefix, name in (("gpt-", "OpenAI"), ("gemini-", "Google"), ("claude-", "Anthropic"),
                         ("qwen", "Qwen"), ("glm-", "ZAI"), ("kimi-", "Moonshot AI")):
        if label.startswith(prefix):
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    classes = np.array(sorted(train.model.astype(str).unique()))
    y = np.searchsorted(classes, train.model.astype(str))
    texts, groups = train.text.astype(str).to_numpy(), train.prompt_id.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(texts, y, groups))
    for fit_idx, valid_idx in splits:
        assert not set(groups[fit_idx]) & set(groups[valid_idx])
    oof_scores = np.full((len(train), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(train), -1, dtype=np.int32)
    fold_rows = []
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        t0 = time.perf_counter()
        X, (V,) = build_features(texts[fit_idx], [texts[valid_idx]])
        clf = LinearSVC(C=0.5, random_state=SEED, max_iter=4000).fit(X, y[fit_idx])
        scores = clf.decision_function(V)
        oof_scores[valid_idx] = scores
        fold_ids[valid_idx] = fold
        pred = scores.argmax(axis=1)
        fold_rows.append({"fold": fold, "balanced_accuracy": float(balanced_accuracy_score(y[valid_idx], pred)),
                          "accuracy": float(accuracy_score(y[valid_idx], pred)),
                          "n_train": int(len(fit_idx)), "n_valid": int(len(valid_idx)),
                          "seconds": time.perf_counter() - t0})
        print(f"fold {fold + 1}/5 BA={fold_rows[-1]['balanced_accuracy']:.6f}", flush=True)
        del X, V, clf
        gc.collect()
    assert np.isfinite(oof_scores).all()
    pred = oof_scores.argmax(axis=1)
    recalls = recall_score(y, pred, labels=np.arange(len(classes)), average=None, zero_division=0)
    true_provider = np.array([provider(classes[i]) for i in y])
    pred_provider = np.array([provider(classes[i]) for i in pred])
    providers = sorted(set(true_provider))
    metrics = {
        "recipe": "char35_styletext__svc_0p5",
        "validation": "5-fold GroupKFold(prompt_id), fold-local TF-IDF",
        "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pooled_oof_accuracy": float(accuracy_score(y, pred)),
        "pooled_oof_macro_f1": float(f1_score(y, pred, average="macro")),
        "folds": fold_rows,
        "per_class_recall": {str(classes[i]): float(recalls[i]) for i in range(len(classes))},
        "provider_recall": {p: float(np.mean(pred_provider[true_provider == p] == p)) for p in providers},
        "features": FEATURES,
        "classifier": {"name": "LinearSVC", "C": 0.5, "max_iter": 4000, "random_state": SEED},
        "train_sha256": hashlib.sha256((data_dir / "train.csv").read_bytes()).hexdigest(),
        "n_train": len(train), "n_test_visible": len(test), "classes": classes.tolist(),
        "elapsed_seconds_cv": time.perf_counter() - start,
    }
    (output_dir / "style_aware_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(output_dir / "style_aware_oof_scores.npy", oof_scores)
    pd.DataFrame({"id": train.id, "prompt_id": train.prompt_id, "fold": fold_ids,
                  "true": train.model, "pred": classes[pred]}).to_csv(output_dir / "style_aware_oof.csv", index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=np.arange(len(classes))), index=classes, columns=classes).to_csv(output_dir / "style_aware_confusion.csv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
