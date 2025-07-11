import logging

from synerise.training_pipeline.tasks import (
    parse_task,
)

from synerise.training_pipeline.train_runner import run_tasks
from synerise.training_pipeline.task_constructor import TaskConstructor
from synerise.training_pipeline.logger_factory import NeptuneLoggerFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_pipeline(embeddings_dir, data_dir=None, task_names=None):
    if task_names is None:
        task_names = ["propensity_sku", "propensity_category"]
    logger.info(f"Running training pipeline with embeddings from {embeddings_dir}")
    tasks = [parse_task(task) for task in task_names]

    task_constructor = TaskConstructor(data_dir=data_dir)

    score_dir = embeddings_dir

    neptune_logger_factory = NeptuneLoggerFactory(
        project=None,
        api_key=None,
        name="optuna_trial",
    )

    try:
        run_tasks(
            neptune_logger_factory=neptune_logger_factory,
            tasks=tasks,
            task_constructor=task_constructor,
            data_dir=data_dir,
            embeddings_dir=embeddings_dir,
            num_workers=8,
            accelerator="gpu",
            devices=[0],
            score_dir=score_dir,
            disable_relevant_clients_check=True,
        )

        scores_path = embeddings_dir / "scores.json"
        if scores_path.exists():
            import json
            with open(scores_path, "r") as f:
                scores = json.load(f)

            logger.info(f"Scores: {scores}")

            if len(scores) > 0:
                avg_score = sum(scores.values()) / len(scores)
                logger.info(f"Average score: {avg_score}")
                return avg_score, scores

        logger.error("No scores found after running training pipeline")
        return 0, {}

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 0, {}