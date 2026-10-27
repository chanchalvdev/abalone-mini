"""Shared settings for every script. Run everything in us-east-1 to keep it simple."""
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROJECT = "abalone-mini"                      # every resource name starts with this
PREFIX = PROJECT                              # S3 folder inside your SageMaker bucket
MODEL_PACKAGE_GROUP = f"{PROJECT}-models"     # Model Registry group
PIPELINE_NAME = f"{PROJECT}-pipeline"
ENDPOINT_NAME = f"{PROJECT}-serverless"

INSTANCE_TYPE = "ml.m5.large"                 # small and cheap; change if you hit a quota error
XGB_VERSION = "1.7-1"                         # SageMaker built-in XGBoost container
SKLEARN_VERSION = "1.2-1"                     # SageMaker scikit-learn container (for processing)

# Public copy of the UCI abalone dataset (headerless CSV, ~4,177 rows)
PUBLIC_BUCKET = "sagemaker-sample-files"
PUBLIC_KEY = "datasets/tabular/uci_abalone/abalone.csv"

# Check the Bedrock "Model catalog" in your region and change this if needed
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")


def boto_session():
    return boto3.Session(region_name=REGION)


def get_role():
    """SageMaker execution role. Inside SageMaker Studio this is found automatically."""
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    from sagemaker.core.helper.session_helper import get_execution_role
    return get_execution_role()


def get_bucket(sess=None):
    """Your SageMaker default bucket (sagemaker-<region>-<account>), unless overridden."""
    bucket = os.environ.get("SAGEMAKER_BUCKET")
    if bucket:
        return bucket
    return sess.default_bucket()


def s3(bucket, *parts):
    return "s3://" + "/".join([bucket, PREFIX, *parts])
