from prefect import task, get_run_logger

from src.training.train import train

@task(name="Model Training")
def task_train():
    """
    Execute the end-to-end churn training and promotion lifecycle.

    The flow prepares and snapshots data, trains a Candidate, evaluates promotion
    gates and, when accepted, publishes and verifies an immutable serving release.

    Args:
        force_run: Bypass the stable-system training skip condition.
        bootstrap: Create the initial Champion in an empty registry.

    Returns:
        Training, promotion, release and deployment metadata, or None when training
        is skipped.
    """
    p_logger = get_run_logger()
    p_logger.info("Triggering model training task.")
    model, run_id = train()
    return run_id