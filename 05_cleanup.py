"""PART 5 - Cleanup. Run this before you close your laptop.

Default: deletes the endpoint, endpoint configs and models (the things that can cost money).
--all:   also deletes the pipeline, the Model Registry versions + group, and S3 files.

Run:  python 05_cleanup.py          or          python 05_cleanup.py --all
"""
import argparse

from sagemaker.core.helper.session_helper import Session

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    boto = config.boto_session()
    sm = boto.client("sagemaker")

    # 1. Endpoint (the only always-on cost if it were real-time)
    try:
        sm.delete_endpoint(EndpointName=config.ENDPOINT_NAME)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=config.ENDPOINT_NAME)
        print("Deleted endpoint", config.ENDPOINT_NAME)
    except sm.exceptions.ClientError:
        print("No endpoint to delete")

    # 2. Endpoint configs and models created by this project
    for cfg in sm.list_endpoint_configs(NameContains=config.PROJECT, MaxResults=100)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print("Deleted endpoint config", cfg["EndpointConfigName"])
    for m in sm.list_models(NameContains=config.PROJECT, MaxResults=100)["Models"]:
        sm.delete_model(ModelName=m["ModelName"])
        print("Deleted model", m["ModelName"])

    if not args.all:
        print("\nDone. Use --all to also remove pipeline, registry and S3 data.")
        return

    # 3. Pipeline
    try:
        sm.delete_pipeline(PipelineName=config.PIPELINE_NAME)
        print("Deleted pipeline", config.PIPELINE_NAME)
    except sm.exceptions.ClientError:
        pass

    # 4. Model Registry
    try:
        pkgs = sm.list_model_packages(ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
                                      MaxResults=100)["ModelPackageSummaryList"]
        for p in pkgs:
            sm.delete_model_package(ModelPackageName=p["ModelPackageArn"])
        sm.delete_model_package_group(ModelPackageGroupName=config.MODEL_PACKAGE_GROUP)
        print(f"Deleted {len(pkgs)} model versions and group", config.MODEL_PACKAGE_GROUP)
    except sm.exceptions.ClientError as e:
        print("Registry cleanup skipped:", e)

    # 5. S3 data under s3://<bucket>/abalone-mini/
    bucket = config.get_bucket(Session(boto_session=boto))
    b = boto.resource("s3").Bucket(bucket)
    deleted = b.objects.filter(Prefix=f"{config.PREFIX}/").delete()
    print("Deleted S3 objects under", config.s3(bucket), f"({len(deleted)} batch)")
    print("\nAlso: stop your SageMaker Studio JupyterLab space so it stops billing.")


if __name__ == "__main__":
    main()
