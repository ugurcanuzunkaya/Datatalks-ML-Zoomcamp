import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the dataset.
    """
    # Check for duplicates
    df = df.drop_duplicates()

    # Missing values (none expected in this dataset, but good practice)
    df = df.dropna()

    return df


def preprocess_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.
    """
    X = df.drop(columns=["charges"])
    y = df["charges"]
    return X, y
