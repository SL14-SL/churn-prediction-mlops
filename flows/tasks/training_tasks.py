from prefect import task, get_run_logger

from src.training.train import train

@task(name="Model Training")
def task_train():
    p_logger = get_run_logger()
    p_logger.info("Triggering model training task.")
    model, run_id = train()
    return run_id