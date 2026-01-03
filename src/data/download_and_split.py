import argparse
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_URL = "https://raw.githubusercontent.com/dumitrescustefan/Romanian-Transformers/examples/examples/sentiment_analysis/ro/train.csv"
TEST_URL = "https://raw.githubusercontent.com/dumitrescustefan/Romanian-Transformers/examples/examples/sentiment_analysis/ro/test.csv"


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    raw = base_dir / "raw"
    processed = base_dir / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, processed


def download_csv(url: str) -> pd.DataFrame:
    # Pandas can read directly from raw GitHub URLs
    df = pd.read_csv(url)
    return df


def stratified_split(df: pd.DataFrame, label_col: str, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df[label_col] if label_col in df.columns else None,
    )
    return train_df, val_df


TEXT_COL = "text"
LABEL_COL = "label"


def main():
    parser = argparse.ArgumentParser(description="Download ro_sent and create stratified train/val split.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory where raw/processed are stored")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio from training set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    raw_dir, processed_dir = ensure_dirs(base_dir)

    print("Downloading training CSV...")
    train_df_full = download_csv(TRAIN_URL)
    print(f"Train rows: {len(train_df_full)}; columns: {list(train_df_full.columns)}")

    print("Downloading test CSV...")
    test_df = download_csv(TEST_URL)
    print(f"Test rows: {len(test_df)}; columns: {list(test_df.columns)}")

    # Persist raw copies
    train_raw_path = raw_dir / "train.csv"
    test_raw_path = raw_dir / "test.csv"
    train_df_full.to_csv(train_raw_path, index=False)
    test_df.to_csv(test_raw_path, index=False)
    print(f"Saved raw: {train_raw_path} and {test_raw_path}")

    print("Creating stratified train/val split...")
    train_df, val_df = stratified_split(train_df_full, LABEL_COL, args.val_size, args.seed)

    train_out = processed_dir / "train.csv"
    val_out = processed_dir / "val.csv"
    test_out = processed_dir / "test.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Saved processed: {train_out}, {val_out}, {test_out}")
    print("Done.")


if __name__ == "__main__":
    main()
