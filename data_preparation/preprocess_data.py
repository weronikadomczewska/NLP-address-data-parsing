from datasets import load_dataset
import pandas as pd
import json
import re
from typing import Any, Dict, Iterator
import numpy as np
from sklearn.model_selection import train_test_split


def _clean_text(text: Any) -> Any:
    """
    Clean control characters and placeholder sequences from text.
    
    Args:
        text (Any): The input text to clean.
        
    Returns:
        Any: The cleaned text if input is a string, otherwise the original input.
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data() -> pd.DataFrame:
    """
    Load the dataset and return a DataFrame with cleaned `input` and `target`.
    Input is a cleaned version of the `user` field. Target is a dictionary containing cleaned 
    values of `house_number`, `street`, `city`, and `country` extracted from the `assistant` field.

    Usage:
        df = preprocess_data() # DataFrame with columns `input` and `target`

    Returns:
        pd.DataFrame: A DataFrame with columns `input` and `target`.
    """
    print("Loading dataset...")
    path = "Josephgflowers/mixed-address-parsing"
    dataset = load_dataset(path)

    df = pd.DataFrame(dataset["train"])
    
    print("Cleaning data...")
    valid_fields = ["house_number", "street", "city", "country", "postal_code", "state"]
    df["target"] = df["assistant"].apply(lambda x: {field: _clean_text(json.loads(x).get(field, "")) for field in valid_fields})
    df["input"] = df["user"].apply(_clean_text)
    df = df[["input", "target"]]
    return df


def make_splits(df: pd.DataFrame) -> Iterator[Dict[str, pd.DataFrame]]:
    """
    Yield deterministic train/val/test index splits per fold.

    Usage:
        for split in make_splits(df):
            train, val, test = split['train'], split['val'], split['test']
            # train on train, eval on val, final test on test

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        train_pct (float): Proportion of data to use for training.
        val_pct (float): Proportion of data to use for validation.
        test_pct (float): Proportion of data to use for testing.
        seed (int): Random seed for reproducibility.
        n_folds (int): Number of folds to generate.

    Yields:
        Dict[str, pd.DataFrame]: A dictionary containing the train, val, and test DataFrames for each fold.
    """
    TRAIN_PCT: float = 0.7
    VAL_PCT: float = 0.15
    TEST_PCT: float = 0.15
    SEED: int = 42
    N_FOLDS: int = 3
    DATA_SIZE = 10_000

    print(f"Downsampling data from {len(df)} to {DATA_SIZE} samples...")
    df = df.sample(n=DATA_SIZE, random_state=SEED).reset_index(drop=True)

    val_rel = VAL_PCT / (TRAIN_PCT + VAL_PCT)

    for fold in range(N_FOLDS):
        rs1 = SEED + fold
        rs2 = SEED + fold + 1000

        train_val, test = train_test_split(
            df, test_size=TEST_PCT, random_state=rs1, shuffle=True
        )

        train, val = train_test_split(
            train_val, test_size=val_rel, random_state=rs2, shuffle=True
        )

        yield {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }


def evaluate_predictions(golds, preds, fields=None) -> Dict[str, Any]:
    """Evaluate predictions against gold targets.

    Args:
        golds: iterable of gold target dicts (e.g., row['target'] values from the DataFrame).
        preds: iterable of model prediction strings or dict-like objects (JSON strings are expected).
        fields: list of fields to evaluate. If None, defaults to
            ['house_number','street','city','country','postal_code','state'].

    Returns:
        A dict with keys:
          - 'exact_match': fraction of examples with exact match across all fields.
          - 'per_field_accuracy': dict mapping field -> accuracy (correct / total).
          - 'overall_item_accuracy': accuracy computed over all individual field items.
          - 'counts': diagnostic counts for examples, per-field totals and corrects.
    """
    if fields is None:
        fields = ["house_number", "street", "city", "country", "postal_code", "state"]

    total_examples = 0
    exact_matches = 0

    field_correct = {f: 0 for f in fields}
    field_total = {f: 0 for f in fields}

    for gold, pred in zip(golds, preds):
        total_examples += 1

        if isinstance(gold, dict):
            g = {f: _clean_text(gold.get(f, "")) or "" for f in fields}
        else:
            g = {f: "" for f in fields}

        p_parsed = None
        if isinstance(pred, str):
            try:
                p_parsed = json.loads(pred)
            except Exception:
                p_parsed = None
        elif isinstance(pred, dict):
            p_parsed = pred

        if isinstance(p_parsed, dict):
            p = {f: _clean_text(p_parsed.get(f, "")) or "" for f in fields}
        else:
            p = {f: "" for f in fields}

        if all(g[f] == p[f] for f in fields):
            exact_matches += 1

        for f in fields:
            field_total[f] += 1
            if g[f] == p[f]:
                field_correct[f] += 1

    overall_item_total = sum(field_total.values())
    overall_item_correct = sum(field_correct.values())

    per_field_accuracy = {
        f: (field_correct[f] / field_total[f]) if field_total[f] else 0.0 for f in fields
    }

    return {
        "exact_match": (exact_matches / total_examples) if total_examples else 0.0,
        "per_field_accuracy": per_field_accuracy,
        "overall_item_accuracy": (overall_item_correct / overall_item_total) if overall_item_total else 0.0,
        "counts": {
            "examples": total_examples,
            "per_field_total": field_total,
            "per_field_correct": field_correct,
        },
    }


