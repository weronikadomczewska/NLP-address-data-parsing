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


