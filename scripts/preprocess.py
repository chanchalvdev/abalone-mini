"""Processing script: raw abalone CSV -> train / validation / test CSVs.

Built-in XGBoost wants: label in the FIRST column, no header row.
Inside a Processing Job, SageMaker copies S3 input to /opt/ml/processing/input
and uploads whatever we write to /opt/ml/processing/<output> back to S3.
"""
import glob
import os

import pandas as pd

COLUMNS = ["sex", "length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight", "rings"]
FEATURES = ["sex_F", "sex_I", "sex_M", "length", "diameter", "height",
            "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]


def prepare(raw: pd.DataFrame, seed: int = 42):
    """One-hot encode sex, put label first, shuffle, split 70/15/15."""
    df = raw.copy()
    df.columns = COLUMNS
    for s in ["F", "I", "M"]:
        df[f"sex_{s}"] = (df["sex"] == s).astype(int)
    df = df[["rings"] + FEATURES]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(0.70 * n)]
    validation = df.iloc[int(0.70 * n): int(0.85 * n)]
    test = df.iloc[int(0.85 * n):]
    return train, validation, test


def main(base="/opt/ml/processing"):
    files = glob.glob(f"{base}/input/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV found in {base}/input")
    raw = pd.read_csv(files[0], header=None)
    splits = dict(zip(["train", "validation", "test"], prepare(raw)))
    for name, part in splits.items():
        os.makedirs(f"{base}/{name}", exist_ok=True)
        part.to_csv(f"{base}/{name}/{name}.csv", header=False, index=False)
        print(f"{name}: {len(part)} rows")


if __name__ == "__main__":
    main()
