"""PART 2 - SageMaker Pipeline: process -> train -> evaluate -> condition -> register.

Flow:
  Preprocess (Processing job, sklearn container)
      -> Train (Training job, built-in XGBoost, Spot)
      -> Evaluate (Processing job, writes evaluation.json = PropertyFile)
      -> Condition: RMSE <= RmseThreshold ?
            yes -> CreateModel + RegisterModel (PendingManualApproval)
            no  -> Fail step (pipeline stops with a clear message)

Run 01_training_job.py first (it copies the raw CSV into your bucket).
Run:  python 02_pipeline.py
"""
import time

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (ProcessingInput, ProcessingOutput,
                                   ProcessingS3Input, ProcessingS3Output)
from sagemaker.core.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.core.workflow.functions import Join, JsonGet
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (CheckpointConfig, Compute, InputData,
                                     OutputDataConfig, StoppingCondition)

import config


def s3_in(name, uri, local):
    return ProcessingInput(input_name=name, s3_input=ProcessingS3Input(
        s3_uri=uri, local_path=local, s3_data_type="S3Prefix",
        s3_input_mode="File", s3_data_distribution_type="FullyReplicated"))


def s3_out(name, uri, local):
    return ProcessingOutput(output_name=name, s3_output=ProcessingS3Output(
        s3_uri=uri, local_path=local, s3_upload_mode="EndOfJob"))


def build_pipeline(role, bucket, boto):
    psess = PipelineSession(boto_session=boto, default_bucket=bucket)
    cache = CacheConfig(enable_caching=True, expire_after="P7D")   # reruns skip unchanged steps

    # ---- Parameters: change per run without editing code -------------------
    input_data = ParameterString("InputDataUrl", default_value=config.s3(bucket, "raw", "abalone.csv"))
    train_instance = ParameterString("TrainingInstanceType", default_value=config.INSTANCE_TYPE)
    rmse_threshold = ParameterFloat("RmseThreshold", default_value=3.0)
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    sklearn_img = image_uris.retrieve(framework="sklearn", region=config.REGION,
                                      version=config.SKLEARN_VERSION, py_version="py3",
                                      instance_type=config.INSTANCE_TYPE)
    xgb_img = image_uris.retrieve(framework="xgboost", region=config.REGION,
                                  version=config.XGB_VERSION)

    # ---- Step 1: Preprocess -------------------------------------------------
    pre = ScriptProcessor(role=role, image_uri=sklearn_img, command=["python3"],
                          instance_type=config.INSTANCE_TYPE, instance_count=1,
                          base_job_name=f"{config.PROJECT}-preprocess", sagemaker_session=psess)
    step_pre = ProcessingStep(
        name="Preprocess",
        step_args=pre.run(
            code="scripts/preprocess.py",
            inputs=[s3_in("raw", input_data, "/opt/ml/processing/input")],
            outputs=[s3_out(n, config.s3(bucket, "pipeline", n), f"/opt/ml/processing/{n}")
                     for n in ("train", "validation", "test")],
        ),
        cache_config=cache,
    )
    outputs = step_pre.properties.ProcessingOutputConfig.Outputs

    # ---- Step 2: Train ------------------------------------------------------
    trainer = ModelTrainer(
        sagemaker_session=psess, role=role, base_job_name=f"{config.PROJECT}-train",
        training_image=xgb_img,
        compute=Compute(instance_type=train_instance, instance_count=1, volume_size_in_gb=10,
                        enable_managed_spot_training=True),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=900,
                                             max_wait_time_in_seconds=1800),
        checkpoint_config=CheckpointConfig(s3_uri=config.s3(bucket, "pipeline", "checkpoints"),
                                           local_path="/opt/ml/checkpoints"),
        output_data_config=OutputDataConfig(s3_output_path=config.s3(bucket, "pipeline", "model")),
        hyperparameters={"objective": "reg:squarederror", "num_round": "150", "max_depth": "5",
                         "eta": "0.2", "subsample": "0.8", "eval_metric": "rmse"},
        input_data_config=[
            InputData(channel_name="train", data_source=outputs["train"].S3Output.S3Uri,
                      content_type="text/csv"),
            InputData(channel_name="validation", data_source=outputs["validation"].S3Output.S3Uri,
                      content_type="text/csv"),
        ],
    )
    step_train = TrainingStep(name="Train", step_args=trainer.train(), cache_config=cache)
    model_s3 = step_train.properties.ModelArtifacts.S3ModelArtifacts

    # ---- Step 3: Evaluate (writes evaluation.json) --------------------------
    evaluator = ScriptProcessor(role=role, image_uri=xgb_img, command=["python3"],
                                instance_type=config.INSTANCE_TYPE, instance_count=1,
                                base_job_name=f"{config.PROJECT}-evaluate", sagemaker_session=psess)
    report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="Evaluate",
        step_args=evaluator.run(
            code="scripts/evaluate.py",
            inputs=[s3_in("model", model_s3, "/opt/ml/processing/model"),
                    s3_in("test", outputs["test"].S3Output.S3Uri, "/opt/ml/processing/test")],
            outputs=[s3_out("evaluation", config.s3(bucket, "pipeline", "evaluation"),
                            "/opt/ml/processing/evaluation")],
        ),
        property_files=[report],
    )
    rmse = JsonGet(step_name=step_eval.name, property_file=report,
                   json_path="regression_metrics.rmse.value")

    # ---- Step 4: Register (only reached if the condition passes) -------------
    builder = ModelBuilder(s3_model_data_url=model_s3, image_uri=xgb_img,
                           sagemaker_session=psess, role_arn=role)
    step_create = ModelStep(name="CreateModel", step_args=builder.build())
    step_register = ModelStep(
        name="RegisterModel",
        step_args=builder.register(
            model_package_group_name=config.MODEL_PACKAGE_GROUP,
            content_types=["text/csv"], response_types=["text/csv"],
            inference_instances=[config.INSTANCE_TYPE], transform_instances=[config.INSTANCE_TYPE],
            approval_status=approval,
            description="Abalone rings regressor (built-in XGBoost)",
        ),
    )
    step_fail = FailStep(name="RmseTooHigh",
                         error_message=Join(on=" ", values=["RMSE is above threshold", rmse_threshold]))

    # ---- Step 5: Condition --------------------------------------------------
    step_cond = ConditionStep(
        name="CheckRmse",
        conditions=[ConditionLessThanOrEqualTo(left=rmse, right=rmse_threshold)],
        if_steps=[step_create, step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=config.PIPELINE_NAME,
        parameters=[input_data, train_instance, rmse_threshold, approval],
        steps=[step_pre, step_train, step_eval, step_cond],
        sagemaker_session=psess,
    )


def main():
    boto = config.boto_session()
    sess = Session(boto_session=boto)
    role = config.get_role()
    bucket = config.get_bucket(sess)
    sm = boto.client("sagemaker")

    pipeline = build_pipeline(role, bucket, boto)
    pipeline.upsert(role_arn=role)                      # create or update the definition
    arn = sm.start_pipeline_execution(PipelineName=config.PIPELINE_NAME)["PipelineExecutionArn"]
    print("Started:", arn)
    print("Watch it live: SageMaker Studio -> Pipelines ->", config.PIPELINE_NAME)

    status = "Executing"
    while status == "Executing":
        time.sleep(30)
        status = sm.describe_pipeline_execution(PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        print("  status:", status)

    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")
    for s in steps["PipelineExecutionSteps"]:
        print(f"  {s['StepName']:<15} {s['StepStatus']:<10} {s.get('FailureReason', '')}")
    if status == "Succeeded":
        print("\nNext: python 03_approve_and_deploy.py --approve")


if __name__ == "__main__":
    main()
