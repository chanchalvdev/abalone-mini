"""PART 3 - Model Registry approval gate -> Serverless endpoint -> live prediction.

An endpoint is 3 objects:  Model (artifact + container)  ->  EndpointConfig (how to host)
->  Endpoint (the live HTTPS API). We use plain boto3 here so you see each one.

Run:  python 03_approve_and_deploy.py --approve
"""
import argparse
import time

import config

SAMPLE = "0,0,1,0.455,0.365,0.095,0.514,0.2245,0.101,0.15"   # sex_F,sex_I,sex_M,length,...,shell_weight


def latest_package(sm, status=None):
    kwargs = dict(ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
                  SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    if status:
        kwargs["ModelApprovalStatus"] = status
    items = sm.list_model_packages(**kwargs)["ModelPackageSummaryList"]
    return items[0] if items else None


def endpoint_exists(sm):
    try:
        sm.describe_endpoint(EndpointName=config.ENDPOINT_NAME)
        return True
    except sm.exceptions.ClientError:
        return False


def predict(runtime, row=SAMPLE):
    resp = runtime.invoke_endpoint(EndpointName=config.ENDPOINT_NAME,
                                   ContentType="text/csv", Body=row)
    return float(resp["Body"].read().decode().strip().split(",")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="approve the newest pending version")
    args = ap.parse_args()

    boto = config.boto_session()
    sm, runtime = boto.client("sagemaker"), boto.client("sagemaker-runtime")
    role = config.get_role()

    # ---- Step 1: approval gate ---------------------------------------------
    newest = latest_package(sm)
    if newest is None:
        raise SystemExit("No model versions yet. Run 02_pipeline.py first.")
    print(f"Newest version: v{newest['ModelPackageVersion']} -> {newest['ModelApprovalStatus']}")
    if newest["ModelApprovalStatus"] == "PendingManualApproval" and args.approve:
        sm.update_model_package(ModelPackageArn=newest["ModelPackageArn"],
                                ModelApprovalStatus="Approved",
                                ApprovalDescription="Approved after RMSE check")
        print("Approved. (In a real team, this event triggers CI/CD via EventBridge.)")

    approved = latest_package(sm, status="Approved")
    if approved is None:
        raise SystemExit("Nothing approved yet. Rerun with --approve.")
    pkg_arn = approved["ModelPackageArn"]
    print("Deploying:", pkg_arn)

    # ---- Step 2: Model -> EndpointConfig -> Endpoint ------------------------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{config.PROJECT}-model-{stamp}"
    cfg_name = f"{config.PROJECT}-cfg-{stamp}"
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role,
                    PrimaryContainer={"ModelPackageName": pkg_arn})
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 2},  # scales to zero
        }],
    )
    if endpoint_exists(sm):
        sm.update_endpoint(EndpointName=config.ENDPOINT_NAME, EndpointConfigName=cfg_name)
        print("Updating existing endpoint (new version, same URL)...")
    else:
        sm.create_endpoint(EndpointName=config.ENDPOINT_NAME, EndpointConfigName=cfg_name)
        print("Creating endpoint (about 3-6 minutes)...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=config.ENDPOINT_NAME,
                                              WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print("Endpoint InService:", config.ENDPOINT_NAME)

    # ---- Step 3: call it ----------------------------------------------------
    t0 = time.time()
    rings = predict(runtime)
    print(f"Predicted rings: {rings:.1f}  (age about {rings + 1.5:.1f} years)  "
          f"latency {1000 * (time.time() - t0):.0f} ms (first call includes cold start)")
    t0 = time.time()
    predict(runtime)
    print(f"Warm call latency: {1000 * (time.time() - t0):.0f} ms")


if __name__ == "__main__":
    main()
