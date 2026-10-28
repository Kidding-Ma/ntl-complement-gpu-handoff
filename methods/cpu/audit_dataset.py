"""Audit a private Name That LLM data directory without publishing its text."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text.strip()).casefold()


def provider(label: str) -> str:
    for prefix, name in (
        ("gpt-", "OpenAI"),
        ("gemini-", "Google"),
        ("claude-", "Anthropic"),
        ("qwen", "Qwen"),
        ("glm-", "ZAI"),
        ("kimi-", "Moonshot AI"),
    ):
        if label.startswith(prefix):
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Private directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Private output directory for compact audit files")
    parser.add_argument("--external-pool", type=Path, default=None,
                        help="Optional private pool parquet for overlap/label conflict checks")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    required_train = ["id", "prompt_id", "text", "model"]
    required_test = ["id", "text"]
    assert list(train.columns) == required_train, train.columns.tolist()
    assert list(test.columns) == required_test, test.columns.tolist()
    assert train.id.is_unique and test.id.is_unique
    assert train[required_train].notna().all().all()
    assert test[required_test].notna().all().all()

    train_norm = train.text.map(norm_text)
    test_norm = test.text.map(norm_text)
    prompt_model = train.groupby(["prompt_id", "model"], dropna=False).size()
    prompt_sizes = train.groupby("prompt_id", dropna=False).size()
    model_counts = train.model.value_counts().sort_index()
    labels = sorted(train.model.astype(str).unique())

    train_texts = set(train.text.astype(str))
    test_texts = set(test.text.astype(str))
    train_norms = set(train_norm)
    test_norms = set(test_norm)

    # Compact duplicate diagnostics; raw text is never written by this script.
    parent = list(range(len(train)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, value in enumerate(train_norm):
        if value in first:
            union(i, first[value])
        else:
            first[value] = i
    components = pd.Series([find(i) for i in range(len(train))])
    component_sizes = components.value_counts()
    groups = components.groupby(components).groups.values()

    external_report = {"checked": False, "reason": "not_requested"}
    if args.external_pool is not None:
        pool = pd.read_parquet(args.external_pool)
        if not {"text", "model"}.issubset(pool.columns):
            raise ValueError("external pool must contain text and model columns")
        pool = pool[["text", "model"]].copy()
        pool["text"] = pool["text"].fillna("").astype(str)
        pool["model"] = pool["model"].astype(str)
        train_text_to_labels = train.groupby(train["text"].astype(str))["model"].agg(lambda s: set(s.astype(str))).to_dict()
        overlap_train = pool["text"].map(lambda x: x in train_text_to_labels)
        conflicts = int(sum(bool(train_text_to_labels.get(text, set()) - {label}) for text, label in zip(pool["text"], pool["model"])))
        external_report = {
            "checked": True,
            "rows": int(len(pool)),
            "labels": sorted(pool["model"].unique().tolist()),
            "empty_text_rows": int((pool["text"].str.strip() == "").sum()),
            "exact_overlap_with_train": int(overlap_train.sum()),
            "conflicting_exact_labels_with_train": conflicts,
            "exact_overlap_with_test": int(pool["text"].isin(set(test["text"].astype(str))).sum()),
            "near_duplicates_checked": False,
        }
        if conflicts:
            raise ValueError("external pool has exact text/label conflicts with official train")

    report = {
        "python": sys.version,
        "data_dir_recorded": False,
        "train": {
            "rows": int(len(train)),
            "columns": train.columns.tolist(),
            "sha256": sha256(train_path),
            "nulls": {k: int(v) for k, v in train.isna().sum().items()},
            "duplicate_ids": int(train.id.duplicated().sum()),
            "duplicate_exact_texts": int(train.text.duplicated().sum()),
            "duplicate_normalized_texts": int(train_norm.duplicated().sum()),
            "normalized_text_unique": int(train_norm.nunique()),
        },
        "test_visible_example": {
            "rows": int(len(test)),
            "columns": test.columns.tolist(),
            "sha256": sha256(test_path),
            "nulls": {k: int(v) for k, v in test.isna().sum().items()},
            "duplicate_ids": int(test.id.duplicated().sum()),
            "text_overlap_exact_with_train": int(len(train_texts & test_texts)),
            "text_overlap_normalized_with_train": int(len(train_norms & test_norms)),
        },
        "labels": labels,
        "label_count": len(labels),
        "class_counts": {str(k): int(v) for k, v in model_counts.items()},
        "provider_counts": {
            str(k): int(v)
            for k, v in train.model.astype(str).map(provider).value_counts().sort_index().items()
        },
        "prompts": {
            "count": int(train.prompt_id.nunique()),
            "size_min": int(prompt_sizes.min()),
            "size_max": int(prompt_sizes.max()),
            "size_counts": {str(k): int(v) for k, v in prompt_sizes.value_counts().sort_index().items()},
            "prompt_model_duplicate_pairs": int((prompt_model > 1).sum()),
            "max_rows_per_prompt_model": int(prompt_model.max()),
            "prompt_with_all_labels": int((train.groupby("prompt_id").model.nunique() == len(labels)).sum()),
        },
        "normalized_duplicate_components": {
            "component_count": int(component_sizes.size),
            "components_size_gt_1": int((component_sizes > 1).sum()),
            "rows_in_components_size_gt_1": int(component_sizes[component_sizes > 1].sum()),
            "max_component_size": int(component_sizes.max()),
            "cross_prompt_components": int(sum(train.loc[idx, "prompt_id"].nunique() > 1 for idx in groups)),
            "cross_model_components": int(sum(train.loc[idx, "model"].nunique() > 1 for idx in groups)),
        },
        "external_pool": external_report,
        "contract": {
            "hidden_test_prompt_id_available": False,
            "submission_columns": ["id", "model"],
            "visible_test_is_hidden_test": False,
            "recommended_validation": "5-fold prompt-grouped CV; fit all vectorizers/scalers inside fold",
            "primary_metric": "pooled 18-class balanced accuracy",
        },
    }

    (output_dir / "verified_data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train.groupby(["prompt_id", "model"]).size().unstack(fill_value=0).to_csv(
        output_dir / "prompt_model_counts.csv"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
