"""Evaluation script (runs as a Processing Job inside the XGBoost container).

Reads model.tar.gz + test.csv, writes evaluation.json.
The pipeline's Condition step reads regression_metrics.rmse.value from that JSON.
"""
import json
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb


def load_booster(path):
    # XGBoost container 1.3-1+ saves with Booster.save_model; older versions used pickle.
    booster = xgb.Booster()
    try:
        booster.load_model(path)
        return booster
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate(model_tar, test_csv, workdir):
    os.makedirs(workdir, exist_ok=True)
    with tarfile.open(model_tar) as tar:
        try:
            tar.extractall(path=workdir, filter="data")
        except TypeError:  # older Python without the filter argument
            tar.extractall(path=workdir)
    booster = load_booster(os.path.join(workdir, "xgboost-model"))

    df = pd.read_csv(test_csv, header=None)
    y_true = df.iloc[:, 0].to_numpy(dtype=float)
    preds = booster.predict(xgb.DMatrix(df.iloc[:, 1:].to_numpy(dtype=float)))

    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    return {
        "regression_metrics": {"rmse": {"value": rmse}, "mae": {"value": mae}},
        "test_rows": int(len(y_true)),
    }


def main(base="/opt/ml/processing"):
    report = evaluate(
        model_tar=f"{base}/model/model.tar.gz",
        test_csv=f"{base}/test/test.csv",
        workdir=f"{base}/extracted",
    )
    os.makedirs(f"{base}/evaluation", exist_ok=True)
    with open(f"{base}/evaluation/evaluation.json", "w") as f:
        json.dump(report, f)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
