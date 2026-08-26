# Google Cloud Production Demo

## Purpose and Scope

This guide describes the cost-conscious Google Cloud deployment used to prove
that the customer-churn training, release and serving lifecycle works outside
the local Docker Compose environment.

It is a production-oriented portfolio deployment, not a continuously operated
enterprise platform.

## Cloud Resources

Terraform provisions:

- Artifact Registry;
- a GCS artifact bucket;
- an MLflow Cloud Run service;
- the `churn-prediction-api` Cloud Run service;
- service accounts and IAM bindings;
- Workload Identity Federation for GitHub Actions.

Prefect Cloud records production training flows. Prometheus, Grafana and
Alertmanager remain part of the local operational demonstration unless a
separate cloud monitoring stack is deployed.

Avoid manually creating a second API service with another name. Terraform and
GitHub Actions must target the same `churn-prediction-api` service.

## Required Local Configuration

The Makefile includes and exports `.env`. Configure at least:

```text
API_KEY=<production-api-key>
PREFECT_API_URL=https://api.prefect.cloud/api/accounts/.../workspaces/...
PREFECT_API_KEY=<prefect-cloud-key>
GCP_PROJECT_ID=<project-id>
GCP_REGION=europe-west1
GCP_BUCKET_NAME=<artifact-bucket>
GCP_ARTIFACT_REPO=<artifact-registry-path>
MLFLOW_UI_URL=https://<mlflow-service>.run.app/
PREDICTION_API_URL=https://<api-service>.run.app/predict
```

Do not commit `.env`, API keys, Prefect keys, service-account credentials or
Slack webhooks.

Validate the values without printing secrets:

```bash
make debug-prod-env
make check-prod-env
```

The validation must reject local MLflow, API and Prefect URLs for production
targets.

Authenticate locally when necessary:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

## Provision Infrastructure

```bash
terraform -chdir=infrastructure init
terraform -chdir=infrastructure fmt -check
terraform -chdir=infrastructure validate
terraform -chdir=infrastructure plan -out=tfplan
terraform -chdir=infrastructure apply tfplan
```

Review replacements and deletions before applying. Confirm Cloud Run names,
IAM targets, bucket operations and Workload Identity conditions.

Terraform plan files, `.terraform/`, state and variable files containing local
values must not be committed.

### Existing Cloud Run resources

If a resource already exists but Terraform does not own it, import it instead
of creating a second service. For the API, the resource address and import ID
must match the current Terraform block.

Run a fresh plan after every import and require `0 to destroy` unless a deletion
is explicitly intended.

## GitHub Actions Configuration

Repository configuration includes:

### Secrets

- `GCP_WIF_PROVIDER`;
- `GCP_SA_EMAIL`;
- `API_KEY`;
- production integration secrets required by the workflow.

### Variables

- `GCP_PROJECT_ID`;
- `GCP_REGION`;
- `GCP_ARTIFACT_REPO`;
- `GCP_BUCKET_NAME`;
- `MLFLOW_URL` or the workflow's configured MLflow variable;
- `PREDICTION_API_URL` where required.

The Workload Identity provider condition must reference the exact GitHub
repository and branch, for example:

```text
attribute.repository == 'SL14-SL/mlops-churn-prediction'
assertion.ref == 'refs/heads/main'
```

A repository-name mismatch causes `unauthorized_client` during federated-token
generation.

## CI/CD Deployment

GitHub Actions performs:

1. Ruff validation;
2. unit and integration tests;
3. API smoke testing;
4. Terraform validation or planning;
5. API and MLflow image builds;
6. Trivy vulnerability scans;
7. Artifact Registry publication;
8. Cloud Run deployment.

Application images use immutable Git SHA tags. The production API should not
remain on the Terraform placeholder image after the deployment workflow.

Inspect deployed images:

```bash
gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='value(spec.template.spec.containers[0].image)'

gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='value(spec.template.spec.containers[0].image)'
```

## Upload Raw Demo Data

```bash
make upload-raw-prod
```

Confirm the object:

```bash
gcloud storage ls \
  "gs://${GCP_BUCKET_NAME}/data/raw/"
```

## Bootstrap Production

A fresh MLflow registry has no Champion and no complete serving release:

```bash
make train-bootstrap-prod
```

The target:

- validates the production environment;
- prepares the temporary MLflow demo service;
- uploads raw churn data;
- runs training through Prefect Cloud;
- registers the first Champion;
- publishes and activates an immutable release;
- reloads and verifies the API.

Do not use bootstrap once a Champion exists in the current MLflow backend.
For subsequent forced candidate runs:

```bash
make train-force-prod
```

Forced training still respects Champion/challenger promotion gates.

## Verify Production

Resolve the API base URL from Terraform:

```bash
API_URL="$(
  terraform -chdir=infrastructure \
    output -raw prediction_api_url
)"
```

Check the services:

```bash
curl -fsS "$MLFLOW_UI_URL/health"
curl -fsS "$API_URL/livez" | jq .
curl -fsS "$API_URL/readyz" | jq .
curl -fsS "$API_URL/health" | jq .
```

Readiness must identify the active release, numeric model version, MLflow run
ID, model URI, decision threshold and loaded feature schema.

Execute authenticated semantic inference:

```bash
make predict-test-prod
```

The response must contain:

- a non-null `customer_id`;
- a finite churn probability between zero and one;
- a customer value;
- a configured retention action;
- finite expected value;
- release and model lineage metadata.

## Inspect Cloud Run and Logs

```bash
gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic,status.conditions)'

gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic,status.conditions)'
```

Inspect recent errors:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND severity>=ERROR' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=100
```

## Cost-Conscious MLflow Configuration

The demo uses:

- one MLflow application instance while active;
- SQLite under `/tmp` as tracking backend;
- GCS for artifacts and serving releases;
- increased memory for MLflow registry and UI operations.

SQLite under `/tmp` is ephemeral. A revision or instance replacement can remove
the registry database even while GCS artifacts remain. A stale serving release
cannot load a registry version that no longer exists.

For continuous operation, use PostgreSQL or Cloud SQL and configure a durable
MLflow backend.

## Infrastructure Drift

Terraform should remain the source of truth for service memory, scaling,
environment variables and names. Reconcile emergency `gcloud run services
update` changes back into Terraform.

Run after any manual change:

```bash
terraform -chdir=infrastructure plan
```

## Teardown

```bash
terraform -chdir=infrastructure plan -destroy
terraform -chdir=infrastructure destroy
```

Review buckets, images and externally managed resources separately. Destruction
of cloud resources or data is irreversible.

## Related Documentation

- [Architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Monitoring and SLOs](monitoring-and-slos.md)
- [Operations runbook](operations-runbook.md)

