"""PART 4 - Bedrock Converse API on top of your own SageMaker model.

Your custom model (SageMaker) gives a number. A foundation model (Bedrock)
explains it in plain English. This is the "use both" story for interviews.

Converse = one request/response shape for every Bedrock model,
so switching models is just changing modelId.

Run:  python 04_bedrock_explain.py
"""
import importlib

import config
from scripts.preprocess import FEATURES

deploy = importlib.import_module("03_approve_and_deploy")


def main():
    boto = config.boto_session()
    runtime = boto.client("sagemaker-runtime")
    bedrock = boto.client("bedrock-runtime")

    rings = deploy.predict(runtime)
    features = dict(zip(FEATURES, deploy.SAMPLE.split(",")))

    resp = bedrock.converse(
        modelId=config.BEDROCK_MODEL_ID,
        system=[{"text": "You explain ML predictions to non-experts in 3 short sentences."}],
        messages=[{
            "role": "user",
            "content": [{"text": (
                f"An XGBoost model predicted {rings:.1f} shell rings for this abalone "
                f"(age is roughly rings + 1.5 years). Features: {features}. "
                "Explain what the prediction means and one reason to double-check it."
            )}],
        }],
        inferenceConfig={"maxTokens": 300, "temperature": 0.2},
    )

    print("Prediction:", round(rings, 1), "rings\n")
    print(resp["output"]["message"]["content"][0]["text"])
    usage = resp["usage"]
    print(f"\nTokens in/out: {usage['inputTokens']}/{usage['outputTokens']} | "
          f"latency {resp['metrics']['latencyMs']} ms | stop: {resp['stopReason']}")


if __name__ == "__main__":
    main()
