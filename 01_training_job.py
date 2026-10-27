"""PART 1 - One training job, end to end (your #1 weak area).

What you will see:
  1. Data goes to S3 (training jobs only read from S3 / EFS / FSx).
  2. A training job is configured: image, role, instance, input channels,
     hyperparameters, output path, stopping condition, Spot + checkpoints.
  3. Lifecycle: Starting -> Downloading -> Training -> Uploading -> Completed.
  4. Output: model.tar.gz in S3. Instances are terminated automatically.
  5. Spot savings = 1 - BillableTimeInSeconds / TrainingTimeInSeconds.

Run:  python 01_training_job.py
"""
import io
import time

import pandas as pd
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (CheckpointConfig, Compute, InputData,
                                     OutputDataConfig, StoppingCondition)

import config
from scripts.preprocess import prepare


def upload_csv(s3_client, df, bucket, key, header=False):
    buf = io.StringIO()
    df.to_csv(buf, header=header, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode())
    return f"s3://{bucket}/{key}"


def main():
    boto = config.boto_session()
    sess = Session(boto_session=boto)
    role = config.get_role()
    bucket = config.get_bucket(sess)
    s3c = boto.client("s3")
    sm = boto.client("sagemaker")

    # ---- Step 1: data -> S3 -------------------------------------------------
    obj = s3c.get_object(Bucket=config.PUBLIC_BUCKET, Key=config.PUBLIC_KEY)
    raw_bytes = obj["Body"].read()
    s3c.put_object(Bucket=bucket, Key=f"{config.PREFIX}/raw/abalone.csv", Body=raw_bytes)
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None)
    train, validation, _ = prepare(raw)
    train_uri = upload_csv(s3c, train, bucket, f"{config.PREFIX}/part1/train/train.csv")
    val_uri = upload_csv(s3c, validation, bucket, f"{config.PREFIX}/part1/validation/validation.csv")
    print("Raw data copied to:", config.s3(bucket, "raw", "abalone.csv"))
    print("Train:", train_uri, "| Validation:", val_uri)

    # ---- Step 2: configure the training job --------------------------------
    image = image_uris.retrieve(framework="xgboost", region=config.REGION,
                                version=config.XGB_VERSION)
    base_job_name = f"{config.PROJECT}-part1"
    trainer = ModelTrainer(
        sagemaker_session=sess,
        role=role,
        base_job_name=base_job_name,
        training_image=image,                      # built-in algorithm, no script needed
        compute=Compute(
            instance_type=config.INSTANCE_TYPE,
            instance_count=1,
            volume_size_in_gb=10,
            enable_managed_spot_training=True,     # can save up to ~90%; may be interrupted
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=900,            # hard cap on training time
            max_wait_time_in_seconds=1800,         # Spot: training + waiting for capacity (must be >= max_runtime)
        ),
        checkpoint_config=CheckpointConfig(
            s3_uri=config.s3(bucket, "part1", "checkpoints"),
            local_path="/opt/ml/checkpoints",      # synced to S3 so a Spot restart resumes
        ),
        output_data_config=OutputDataConfig(s3_output_path=config.s3(bucket, "part1", "output")),
        hyperparameters={
            "objective": "reg:squarederror",
            "num_round": "150",
            "max_depth": "5",
            "eta": "0.2",
            "subsample": "0.8",
            "eval_metric": "rmse",
        },
        input_data_config=[
            InputData(channel_name="train", data_source=train_uri, content_type="text/csv"),
            InputData(channel_name="validation", data_source=val_uri, content_type="text/csv"),
        ],
    )

    # ---- Step 3: run it (blocks and streams CloudWatch logs) ----------------
    started = time.time()
    trainer.train(wait=True, logs=True)
    print(f"\nWall clock: {time.time() - started:.0f}s")

    # ---- Step 4: inspect what happened -------------------------------------
    jobs = sm.list_training_jobs(NameContains=base_job_name, SortBy="CreationTime",
                                 SortOrder="Descending", MaxResults=1)["TrainingJobSummaries"]
    job = sm.describe_training_job(TrainingJobName=jobs[0]["TrainingJobName"])
    print("\nJob name:", job["TrainingJobName"], "| Status:", job["TrainingJobStatus"])
    print("Lifecycle:")
    for t in job.get("SecondaryStatusTransitions", []):
        print(f"  {t['Status']:<12} {t.get('StatusMessage', '')}")
    print("Model artifact:", job["ModelArtifacts"]["S3ModelArtifacts"])

    train_s = job.get("TrainingTimeInSeconds", 0)
    bill_s = job.get("BillableTimeInSeconds", 0)
    if train_s:
        print(f"Training time {train_s}s, billable {bill_s}s -> Spot saving {100 * (1 - bill_s / train_s):.0f}%")
    for m in job.get("FinalMetricDataList", []):
        print(f"Metric {m['MetricName']}: {m['Value']:.3f}")

    print("\nCleanup note: the training instance is already gone (SageMaker terminates it).")
    print("Only S3 objects remain; 05_cleanup.py --all removes them.")


if __name__ == "__main__":
    main()
