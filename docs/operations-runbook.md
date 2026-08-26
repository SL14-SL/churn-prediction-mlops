# Churn Prediction API – Incident Response Runbook

## 1. Purpose

This runbook describes how to diagnose and respond to incidents affecting the
customer-churn platform locally or in the Google Cloud production demo.

It covers:

- Churn Prediction API;
- serving releases and model artifacts;
- MLflow;
- Prometheus and Alertmanager;
- Prefect training and deployment flows.

A serving-release rollback is appropriate only when the incident is related to
the active model bundle. Infrastructure, networking and capacity failures are
normally resolved elsewhere.

## 2. Severity Levels

| Severity | Meaning | Examples |
|---|---|---|
| SEV-1 | Decisions completely unavailable | API unavailable, no active bundle |
| SEV-2 | Predictions partially failing or severely degraded | High 5xx rate, major latency increase |
| SEV-3 | Degradation without immediate outage | Monitoring gap, isolated input failures |

## 3. Select the Incident Environment

### 3.1 Local Docker environment

```bash
export INCIDENT_ENV="local"
export API_BASE_URL="http://localhost:8000"
export PROMETHEUS_URL="http://localhost:9090"
```

### 3.2 Google Cloud production demo

Load the project environment without printing secrets:

```bash
set -a
source .env
set +a

export INCIDENT_ENV="prod"
export API_BASE_URL="${PREDICTION_API_URL%/predict}"
export PROMETHEUS_URL="http://localhost:9090"
```

The default production demo is observed by the local Prometheus stack. Keep the
local Prometheus URL unless cloud monitoring was deployed separately.

Validate context:

```bash
printf 'Environment: %s\nAPI: %s\nPrometheus: %s\n' \
  "$INCIDENT_ENV" \
  "$API_BASE_URL" \
  "$PROMETHEUS_URL"

test -n "$API_BASE_URL"
test -n "$PROMETHEUS_URL"
```

Never put API keys, Prefect keys, webhook URLs or credentials in an incident
report.

## 4. General Initial Diagnosis

### 4.1 Check service status

Local:

```bash
docker compose ps
```

Production:

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

### 4.2 Check liveness, readiness and health

```bash
curl -i "$API_BASE_URL/livez"
curl -i "$API_BASE_URL/readyz"
curl -i "$API_BASE_URL/health"
```

Interpretation:

- `/livez = 200`: API process is running.
- `/readyz = 200`: a complete validated bundle is active.
- `/livez = 200`, `/readyz = 503`: process is alive but cannot serve safely.
- all endpoints unreachable: process, container, route or network failure.

### 4.3 Record active lineage

```bash
curl -fsS "$API_BASE_URL/readyz" \
  | jq '{release_id, model_name, model_version, model_run_id, model_uri, serving_alias, decision_threshold}'
```

### 4.4 Inspect logs

Local:

```bash
docker compose logs --since=30m --tail=500 api
```

Production:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="churn-prediction-api"' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=500 \
  --order=asc \
  --format='value(timestamp,severity,httpRequest.status,textPayload,jsonPayload.message)'
```

### 4.5 Inspect active alerts

```bash
curl -s "$PROMETHEUS_URL/api/v1/alerts" \
  | jq '.data.alerts[] | {
      alert: .labels.alertname,
      state: .state,
      severity: .labels.severity,
      instance: .labels.instance,
      active_at: .activeAt
    }'
```

## 5. ChurnAPIUnavailable

### Meaning

Prometheus cannot scrape the API or clients cannot reach it.

### Diagnosis

```bash
curl -i "$API_BASE_URL/livez"
```

Local:

```bash
docker compose ps api
docker compose logs --since=30m --tail=500 api
docker compose ps db mlflow prefect
docker compose exec -T api \
  python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/livez').read().decode())"
```

Production:

```bash
gcloud run revisions list \
  --service churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"

gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format=json | jq '.status.traffic'
```

### Immediate actions

Local:

```bash
docker compose up -d api
docker compose restart api
```

Production Cloud Run has no restart command. Diagnose the revision and deploy a
known-good application image through CI/CD. If necessary, return traffic to a
recorded known-good revision:

```bash
gcloud run services update-traffic churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --to-revisions '<known-good-revision>=100'
```

Record current traffic before changing it and reconcile emergency changes with
CI/CD and Terraform.

### Rollback criteria

A model rollback is normally not appropriate when `/livez` fails. Consider it
only when startup logs prove that loading the newly activated release caused the
failure and the previous release is known to work.

### Recovery verification

```bash
curl -fsS "$API_BASE_URL/livez"
curl -fsS "$API_BASE_URL/readyz" | jq .
```

Run `make predict-test` locally or `make predict-test-prod` in production.

## 6. ChurnServingBundleNotReady

### Meaning

The process is alive, but no complete bundle is active.

Potential causes:

- empty or lost MLflow registry;
- missing active release pointer;
- invalid manifest;
- missing feature schema or prediction probe;
- checksum failure;
- unavailable numeric model version;
- MLflow or GCS IAM failure.

### Diagnosis

```bash
curl -fsS "$API_BASE_URL/readyz" | jq .
```

Local release inspection:

```bash
jq . models/active_serving_release.json
find models/serving_releases -maxdepth 2 -type f | sort
```

Production object inspection:

```bash
gcloud storage ls \
  "gs://${GCP_BUCKET_NAME}/models/serving_releases/"

gcloud storage cat \
  "gs://${GCP_BUCKET_NAME}/models/active_serving_release.json" \
  | jq .
```

Inspect the selected manifest with `gcloud storage cat` and compare its numeric
model version with the MLflow registry.

### Immediate actions

If the active release is complete and accessible, reload it:

```bash
curl -fsS -X POST \
  -H "X-API-KEY: ${API_KEY}" \
  "$API_BASE_URL/admin/reload-model" \
  | jq .
```

Never modify a published release in place.

### Rollback criteria

Roll back when the active release cannot load and a previous release was
verified successfully:

```bash
curl -fsS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{"release_id":"<previous-release-id>"}' \
  "$API_BASE_URL/admin/rollback-serving-release" \
  | jq .
```

### Recovery verification

```bash
curl -fsS "$API_BASE_URL/readyz" \
  | jq '{status, release_id, model_version, model_run_id}'
```

Then execute the environment-appropriate prediction test.

Escalate if multiple releases are damaged, the previous release also fails, or
MLflow/GCS access remains unavailable.

## 7. ChurnPredictionServerErrorRateHigh

### Meaning

More than the configured percentage of `/predict` requests return HTTP 5xx
responses during a window with sufficient traffic.

### Diagnosis

```bash
curl -s "$PROMETHEUS_URL/api/v1/alerts" \
  | jq '.data.alerts[] |
      select(.labels.alertname == "ChurnPredictionServerErrorRateHigh")'

curl -fsS "$API_BASE_URL/readyz" \
  | jq '{release_id, model_version, model_run_id}'
```

Local logs:

```bash
docker compose logs --since=30m api \
  | grep -E 'Prediction failed|ERROR|Traceback'
```

Status metrics:

```bash
curl -sG \
  --data-urlencode 'query=sum by (status_code) (increase(api_response_status_total{path="/predict"}[10m]))' \
  "$PROMETHEUS_URL/api/v1/query" \
  | jq .
```

Run a representative request:

```bash
if [ "$INCIDENT_ENV" = "prod" ]; then
  make predict-test-prod
else
  make predict-test
fi
```

### Immediate actions

- Identify failing payloads and input categories.
- Distinguish request-validation failures from server exceptions.
- Compare failures with the active release time.
- Inspect feature alignment, probability output and decision processing.
- Check MLflow/GCS access only if the failure involves reload or startup.

### Rollback criteria

Rollback is appropriate if errors began after a release, the same request works
with the previous release, or the active bundle has inconsistent lineage.

Do not roll back for invalid client payloads, unrelated resource exhaustion or
network failures.

## 8. ChurnPredictionLatencyHigh

### Meaning

The p95 latency of `/predict` exceeds the configured SLO with sufficient
traffic.

### Diagnosis

```bash
curl -sG \
  --data-urlencode 'query=histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket{path="/predict"}[5m])))' \
  "$PROMETHEUS_URL/api/v1/query" \
  | jq .

curl -sG \
  --data-urlencode 'query=sum(increase(api_request_latency_seconds_count{path="/predict"}[15m]))' \
  "$PROMETHEUS_URL/api/v1/query" \
  | jq .
```

One batch request is one histogram observation regardless of row count.

Local resources:

```bash
docker stats --no-stream
docker compose logs --since=30m api \
  | grep -E 'timing_ms|Prediction completed'
```

Production resources:

```bash
gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(template.scaling,template.containers.resources,status.traffic)'
```

Compare response `metadata.timing_ms.total` with Prometheus end-to-end latency.
A large difference suggests cold-start, network or platform overhead; a high
internal value indicates feature, model or decision processing.

### Immediate actions

- Check batch size and traffic spikes.
- Check CPU and memory.
- Distinguish cold starts from warm latency.
- Confirm that inference does not fetch MLflow artifacts per request.
- Identify the slow timing stage.
- Scale resources only after identifying the bottleneck.

### Rollback criteria

Rollback only when latency increased directly after a model release and the
previous model is demonstrably faster under comparable requests.

## 9. ML Quality or Business Degradation

### Symptoms

- persistent F1, recall or ROC AUC breach;
- excessive Brier score;
- sustained feature drift;
- unexpected action distribution;
- negative or implausible campaign value.

### Diagnosis

- Confirm enough delayed labels are available.
- Inspect class balance and join coverage.
- Verify that the decision threshold matches `/readyz` and the release manifest.
- Compare Champion and Challenger on the same validation data.
- Inspect whether the issue is model quality or decision-policy configuration.

### Actions

Run the automatic policy once:

```bash
make auto-retrain
```

Use `make train-force` or `make train-force-prod` only when candidate training is
intentionally requested. Forced training still does not force promotion.

Do not manually change the active threshold without publishing a new validated
release.

## 10. Prefect or Training Incident

### Local checks

```bash
make prefect-status
docker compose logs --since=30m prefect
```

### Production checks

Confirm that output links to Prefect Cloud rather than `localhost`. Validate
non-secret production values:

```bash
make debug-prod-env
```

If a production run unexpectedly uses local Prefect, verify that the Makefile
uses `LOCAL_PREFECT_API_URL` only for local targets and explicitly forwards
`PREFECT_API_URL` and `PREFECT_API_KEY` in production targets.

Do not repeat a successful bootstrap. Use the normal production training target
after a Champion exists.

## 11. Incident Closure

Close an incident only when:

- `/livez`, `/readyz` and `/health` are successful;
- representative prediction succeeds;
- active release lineage is recorded;
- affected alerts return to `inactive`;
- metrics remain stable for at least one alert window;
- emergency Cloud Run changes are reconciled with Terraform and CI/CD.

Document:

- environment;
- start and end time;
- alert and severity;
- Cloud Run revision where applicable;
- release ID before and after remediation;
- model version and run ID;
- root cause;
- actions and verification evidence;
- application-revision or model-release rollback;
- follow-up owner and deadline.

## Related Documentation

- [System architecture](architecture.md)
- [Local development](local-development.md)
- [Google Cloud production demo](production-demo.md)
- [Serving releases](serving-releases.md)
- [Monitoring, SLOs and alerting](monitoring-and-slos.md)

