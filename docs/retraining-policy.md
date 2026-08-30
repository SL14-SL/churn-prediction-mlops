# Retraining Policy

## Purpose

The retraining policy decides whether new evidence justifies candidate
training. Scheduling, training and promotion remain separate decisions:

- Prefect determines when the policy is evaluated.
- The policy determines whether candidate training should begin.
- Champion/challenger evaluation determines whether the candidate is promoted.

This prevents unconditional retraining and unconditional deployment.

## Evidence Model

| Evidence | Meaning |
|---|---|
| Dataset version | Stable identity of the available training data |
| New labeled rows | Ground-truth rows not consumed by an earlier successful lifecycle |
| Batch IDs | Deterministic identities used for deduplication |
| Data quality | Whether new labels and features are safe to use |
| Performance degradation | Persistent classification-metric breach |
| Feature drift | Persistent distribution change across recent windows |
| Scheduled refresh due | Maximum model age reached |
| Cooldown | Recent training blocks another run |
| Budget availability | Configured workload limits remain respected |

The decision contains an action, stable decision ID, trigger types,
human-readable reasons and the evidence snapshot.

## Decision Order

Blocking gates are evaluated before positive triggers:

1. enough new labeled rows must exist;
2. data quality must pass;
3. workload budget must be available;
4. cooldown must be inactive;
5. at least one valid trigger must be active.

Valid triggers include:

- persistent classification-performance degradation;
- persistent feature drift;
- scheduled model refresh.

If all gates pass, the action is `train_candidate`. Otherwise it is `skip`.

## Classification Performance Evidence

The policy evaluates recent delayed-label windows using classification metrics
appropriate for churn:

- F1 score;
- recall;
- ROC AUC;
- Brier score;
- labeled sample count;
- class balance.

Typical configured gates include minimum F1, recall and ROC AUC plus a maximum
Brier score. Configuration is authoritative; documentation examples are not
hard-coded runtime values.

Windows containing only one target class cannot provide every classification
metric. Such windows are treated as incomplete evidence, not automatic failure.

## Persistence Across Windows

A single noisy window should not trigger training. Performance degradation and
feature drift must remain active across the configured number of consecutive
windows.

Reasons may include:

- insufficient recent windows;
- missing monitoring columns;
- thresholds not breached consecutively;
- too few joined predictions and labels;
- no positive or negative examples in the current window.

Missing evidence is reported rather than interpreted as degradation.

## Ground-Truth Deduplication

Every delayed-label batch receives a deterministic ID. Retraining state records
the batch IDs consumed by a successful lifecycle.

On the next evaluation:

- processed batches remain valid historical evidence;
- their rows are not counted as new training rows;
- only unseen IDs contribute to the minimum-new-data gate.

A skipped decision does not consume batches.

## Cooldown

Cooldown prevents repeated training immediately after a recent run. Its
reference time is resolved from retraining state and successful lifecycle
metadata using timezone-aware UTC timestamps.

Missing or malformed state must not falsely activate cooldown.

## Scheduled Refresh

A maximum model age may request training even without drift or measured
degradation. This ensures that accumulated labels can periodically be
incorporated when monitoring signals are sparse.

Scheduled refresh permits candidate training but does not guarantee promotion.

## Example Decisions

### Insufficient new labels

```json
{
  "action": "skip",
  "reasons": ["Insufficient new training rows."],
  "trigger_types": []
}
```

### Cooldown active

```json
{
  "action": "skip",
  "reasons": ["Retraining cooldown is active."],
  "trigger_types": []
}
```

### Persistent degradation

```json
{
  "action": "train_candidate",
  "reasons": ["Persistent classification performance degradation detected."],
  "trigger_types": ["performance_degradation"]
}
```

## Promotion Remains Independent

`train_candidate` only means that sufficient evidence exists to spend resources
on candidate training. It does not imply that the resulting model will replace
the Champion.

When recent labeled production data is available, the Challenger is evaluated
primarily on that recent production window. Promotion requires:

- a configurable realized-business-value improvement;
- F1 non-regression within the configured tolerance;
- recall non-regression within the configured tolerance;
- ROC-AUC non-regression within the configured tolerance;
- Brier-score non-regression within the configured tolerance.

A separate reference-validation evaluation protects against excessive
regression on the original data distribution. Its F1, recall, ROC-AUC and Brier
score gates act as safety constraints rather than as the primary optimization
objective.

All promotion thresholds are configuration-driven. A candidate may therefore
improve recent production performance and business value while still being
rejected by a reference safety gate.

After the gates pass, the lifecycle performs:

1. model registration and Champion alias assignment;
2. immutable serving-release publication;
3. API reload;
4. readiness and semantic prediction verification.

If evaluation or deployment cannot be completed safely, promotion is blocked
or the previous serving release is restored.

The decision threshold stored in the accepted release belongs to that exact
model version. It must not be independently replaced during deployment.

## Controlled Policy Experiments

Two local controlled experiments demonstrate the distinction between
retraining and promotion:

- the customer-cohort shift uses unchanged real features and labels while
  changing the customer composition; training can occur without promotion;
- the controlled concept-drift experiment applies an explicitly documented and
  audited target-label shift to a defined cohort and demonstrates successful
  adaptive promotion.

Both branches of an experiment use the same ordered customer sequence and
effective labels. Stored hashes and audit artifacts verify that differences
between branches come from the retraining policy rather than different input
data.

```bash
make churn-cohort-shift-comparison
make churn-cohort-shift-comparison-plot

make churn-concept-drift-comparison
make churn-concept-drift-comparison-plot
```
Artifacts are written below:
`results/churn_retraining_comparison/`


## Running the Policy

Evaluate the automatic retraining flow once:

```bash
make auto-retrain
```

Register recurring local execution:

```bash
make prefect-pool
make prefect-setup
make prefect-worker
```

Production training targets use Prefect Cloud credentials from `.env`. Local
targets explicitly use the local Prefect server.

## Lifecycle Demonstration

```bash
make demo-churn-lifecycle
```

The demo simulates inference, label delay, monitoring refresh and retraining
decisions while preserving prediction and model lineage.

## Related Documentation

- [Architecture](architecture.md)
- [Local development](local-development.md)
- [Serving releases](serving-releases.md)
- [Monitoring and SLOs](monitoring-and-slos.md)

