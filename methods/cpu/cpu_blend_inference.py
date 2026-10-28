"""CPU candidate: blend the reviewed style-aware SVC with the original SVC.

The local OOF probe found that averaging row-softmaxed decision scores from
these two different views was more stable than either model alone:
- style-aware four-view LinearSVC(C=0.5): measured grouped OOF BA 0.384484
- original char+word+scalar-style LinearSVC(C=1): measured grouped OOF BA 0.370508
- fold-held-out softmax blend (style weight around 0.70-0.725): 0.393372

This is a candidate, not a hidden-test guarantee. It does not use prompt IDs,
assignment, test quotas, hidden labels, or manual predictions.

Examples:
  python cpu_blend_inference.py --data-dir /private/name-that-llm --output /private/artifacts/dry-run.parquet --dry-run
  python cpu_blend_inference.py --data-dir /private/name-that-llm --output /private/artifacts/submission.parquet
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

STYLE_VIEWS = [
    ("char", 3, 5, False, True, 250000, 1.0),
    ("word", 1, 3, False, True, 180000, 0.7),
    ("shape", 2, 6, False, True, 60000, 0.5),
    ("function", 1, 3, False, True, 50000, 0.5),
]


def shape_text(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_text(text: str) -> str:
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def style_view(text: str, analyzer: str) -> str:
    if analyzer == "shape":
        return shape_text(text)
    if analyzer == "function":
        return function_text(text)
    return str(text)


def build_style_features(train_texts, other_texts):
    fit_parts, other_parts = [], [[] for _ in other_texts]
    for analyzer, lo, hi, lowercase, sublinear, maximum, weight in STYLE_VIEWS:
        actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
        options = dict(
            analyzer=actual, ngram_range=(lo, hi), lowercase=lowercase,
            sublinear_tf=sublinear, min_df=2, max_features=maximum, dtype=np.float64,
        )
        if actual == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        fit_view = [style_view(x, analyzer) for x in train_texts]
        fit_parts.append(vectorizer.fit_transform(fit_view) * weight)
        for j, texts in enumerate(other_texts):
            other_parts[j].append(vectorizer.transform([style_view(x, analyzer) for x in texts]) * weight)
    return hstack(fit_parts, format="csr"), [hstack(x, format="csr") for x in other_parts]


def scalar_style(text: str):
    text = str(text)
    length = max(len(text), 1)
    words = text.split()
    return [
        np.log1p(len(text)), np.log1p(len(words)), text.count("\n") / length * 100,
        text.count("\n\n"),
        *[text.count(ch) for ch in ["—", "–", ",", ".", ";", ":", '"', "'", "’", "“", "*", "#", "(", "-"]],
        sum(c.isupper() for c in text) / length,
        np.mean([len(w) for w in words]) if words else 0.0,
        len(re.findall(r"[.!?]", text)) / max(len(words), 1),
        *[text.count(ch) for ch in ["!", "?", "…", "..."]],
    ]


def build_original_features(train_texts, other_texts):
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 5), sublinear_tf=True,
        lowercase=False, min_df=2, max_features=300000,
    )
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
        lowercase=False, min_df=2, token_pattern=r"\S+",
    )
    scalar_train = np.asarray([scalar_style(t) for t in train_texts], dtype=np.float64)
    scaler = StandardScaler().fit(scalar_train)
    A = hstack([
        char.fit_transform(train_texts), word.fit_transform(train_texts),
        scaler.transform(scalar_train) * 0.3,
    ], format="csr")
    Bs = []
    for texts in other_texts:
        scalar_other = np.asarray([scalar_style(t) for t in texts], dtype=np.float64)
        Bs.append(hstack([
            char.transform(texts), word.transform(texts),
            scaler.transform(scalar_other) * 0.3,
        ], format="csr"))
    return A, Bs


def row_softmax(decision):
    z = np.asarray(decision, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Private directory containing train.csv and test.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Private output submission.parquet path")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--style-weight", type=float, default=0.70,
                        help="Weight on row-softmaxed style-aware SVC; fold-held-out probe preferred 0.70-0.725")
    parser.add_argument("--dry-run", action="store_true", help="Do not write parquet; print predictions and diagnostics")
    args = parser.parse_args()
    if not 0.0 <= args.style_weight <= 1.0:
        raise ValueError("--style-weight must be in [0,1]")
    data_dir = args.data_dir.resolve()
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if len(test) < 50 and not args.dry_run:
        raise ValueError("Visible test is a smoke fixture; use --dry-run for local format checks")
    required_train = {"id", "prompt_id", "text", "model"}
    required_test = {"id", "text"}
    if not required_train.issubset(train.columns) or not required_test.issubset(test.columns) or "model" in test.columns:
        raise ValueError("Unexpected train/test schema")
    if not train["id"].is_unique or not test["id"].is_unique:
        raise ValueError("Duplicate IDs in train/test")
    train_texts = train.text.astype(str).tolist()
    test_texts = test.text.astype(str).tolist()
    labels = np.array(sorted(train.model.astype(str).unique()), dtype=object)
    y = np.searchsorted(labels, train.model.astype(str).to_numpy())
    start = time.perf_counter()

    style_X, (style_T,) = build_style_features(train_texts, [test_texts])
    style_clf = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(style_X, y)
    style_dec = style_clf.decision_function(style_T)

    original_X, (original_T,) = build_original_features(train_texts, [test_texts])
    original_clf = LinearSVC(C=1.0, random_state=42, max_iter=5000).fit(original_X, y)
    original_dec = original_clf.decision_function(original_T)

    # Align columns defensively in case a future sklearn version orders classes differently.
    def align(clf, dec):
        out = np.zeros((len(dec), len(labels)), dtype=np.float64)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = dec[:, j]
        return out
    style_p = row_softmax(align(style_clf, style_dec))
    original_p = row_softmax(align(original_clf, original_dec))
    blend_p = args.style_weight * style_p + (1.0 - args.style_weight) * original_p
    pred = labels[blend_p.argmax(axis=1)]
    sub = pd.DataFrame({"id": test.id.astype(str), "model": pred.astype(str)})[["id", "model"]]
    assert len(sub) == len(test) and sub.id.is_unique and sub.model.isin(labels).all()
    print("style predictions:", labels[style_p.argmax(1)].tolist())
    print("original predictions:", labels[original_p.argmax(1)].tolist())
    print("blend predictions:", pred.tolist())
    print("blend weight:", args.style_weight, "rows:", len(sub), "seconds:", round(time.perf_counter() - start, 2))
    if not args.dry_run:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(output, index=False)
        reread = pd.read_parquet(output)
        pd.testing.assert_frame_equal(sub, reread)
        metadata = args.metadata.resolve() if args.metadata else output.with_suffix(".json")
        metadata.write_text(json.dumps({
            "recipe": "row_softmax_blend(style_aware_svc, original_svc)",
            "style_weight": args.style_weight,
            "original_weight": 1.0 - args.style_weight,
            "data_dir_recorded": True, "output_path_recorded": True,
            "data_dir": str(data_dir), "output_path": str(output),
            "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
            "output_sha256": sha256(output),
            "rows": len(sub), "columns": sub.columns.tolist(),
            "uses_prompt_id_for_inference": False,
            "uses_test_grouping_or_assignment": False,
            "seconds": time.perf_counter() - start,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", output)


if __name__ == "__main__":
    main()
