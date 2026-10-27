# abalone-mini

A small, low-cost, end-to-end **AWS machine learning** project you can finish in one day.
It trains an XGBoost model that predicts an abalone's age from its physical measurements,
then takes that model through the full MLOps lifecycle on **Amazon SageMaker** and adds a
plain-English explanation layer with **Amazon Bedrock**.

```
Training job (Spot + checkpoints)
   -> SageMaker Pipeline (preprocess -> train -> evaluate -> RMSE gate -> register)
   -> Model Registry approval
   -> Serverless endpoint (live predictions)
   -> Bedrock Converse explains the prediction
   -> Cleanup
```

## What you will learn

| Part | Script | Concepts |
|---|---|---|
| 1 | `01_training_job.py` | S3 input channels, built-in XGBoost container, managed Spot training, checkpoints, job lifecycle, Spot savings |
| 2 | `02_pipeline.py` | Processing jobs, pipeline parameters, step caching, `PropertyFile` + `JsonGet`, condition and fail steps, Model Registry |
| 3 | `03_approve_and_deploy.py` | Manual approval gate, Model -> EndpointConfig -> Endpoint, serverless inference, cold vs. warm latency |
| 4 | `04_bedrock_explain.py` | Bedrock Converse API, combining a custom model with a foundation model |
| 5 | `05_cleanup.py` | Deleting every billable resource the project created |

## Project structure

```
abalone-mini/
├── config.py                  # region, resource names, instance type, Bedrock model ID
├── 01_training_job.py         # Part 1: one training job
├── 02_pipeline.py             # Part 2: SageMaker Pipeline
├── 03_approve_and_deploy.py   # Part 3: approve + serverless endpoint
├── 04_bedrock_explain.py      # Part 4: Bedrock explanation
├── 05_cleanup.py              # Part 5: tear everything down
├── scripts/
│   ├── preprocess.py          # one-hot encode, label first, 70/15/15 split
│   └── evaluate.py            # computes RMSE / MAE -> evaluation.json
└── requirements.txt
```

## Dataset

The [UCI Abalone dataset](https://archive.ics.uci.edu/dataset/1/abalone) (~4,177 rows), read from
the public `sagemaker-sample-files` bucket. The target is `rings`; age ≈ rings + 1.5 years.

## Prerequisites

- An AWS account with permissions for SageMaker, S3 and Bedrock
- Region **us-east-1** (set `AWS_REGION` to use another)
- Quota for `ml.m5.large` training/processing (Spot or on-demand)
- Access to **Amazon Nova Lite** in Bedrock (or set `BEDROCK_MODEL_ID` to another model)

## Setup (once)

1. In the AWS console, switch to **us-east-1**.
2. **SageMaker AI** -> set up a domain (quick setup) -> open **Studio** -> start a **JupyterLab** space (`ml.t3.medium`).
3. **Bedrock** -> Model catalog -> confirm Amazon Nova Lite is usable.
4. Clone this repo inside the JupyterLab space, open a terminal in it, then:

```bash
git clone https://github.com/chanchalvdev/abalone-mini.git
cd abalone-mini
pip install -r requirements.txt
```

Running outside Studio? Configure AWS credentials and set `SAGEMAKER_ROLE_ARN` to a role SageMaker can assume.

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region for every resource |
| `SAGEMAKER_ROLE_ARN` | Studio execution role | Role used by jobs and endpoints |
| `SAGEMAKER_BUCKET` | `sagemaker-<region>-<account>` | S3 bucket for data and artifacts |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-lite-v1:0` | Model used in Part 4 |

## Run order

```bash
python 01_training_job.py                   # Part 1: one training job, Spot + checkpoints
python 02_pipeline.py                       # Part 2: full pipeline, waits until done (~10-20 min)
python 03_approve_and_deploy.py --approve   # Part 3: approve + serverless endpoint
python 04_bedrock_explain.py                # Part 4: Bedrock explains the prediction
python 05_cleanup.py --all                  # Part 5: delete everything
```

Run Part 1 before Part 2, because it copies the raw CSV into your bucket.

## Cost

Designed to stay cheap: short Spot training jobs, small processing instances, and a serverless
endpoint that scales to zero. The main costs come from **forgetting to clean up**:

- Always run `python 05_cleanup.py --all` when you are done.
- **Stop your SageMaker Studio JupyterLab space**; it bills while running.

## Troubleshooting

- **`ResourceLimitExceeded`**: your account has no quota for `ml.m5.large`. Change `INSTANCE_TYPE`
  in `config.py` (e.g. `ml.m5.xlarge`) or request a quota increase.
- **`ImportError` from `sagemaker`**: these scripts use SageMaker Python SDK **v3**. Run
  `pip install -U -r requirements.txt`.
- **`AccessDeniedException` from Bedrock**: the role needs `bedrock:InvokeModel`, and the model must
  be enabled in your account.
- **Pipeline failed at `RmseTooHigh`**: the condition step worked as designed. Raise the
  `RmseThreshold` pipeline parameter or improve the model.

## Author

Built by [Chanchal](https://github.com/chanchalvdev).
