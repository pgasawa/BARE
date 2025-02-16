import pickle
from typing import Dict, List

import src.logger as core
from src.tasks import ClassificationEvalTask
from experiments.evals.classification_eval_util import load_val_test_data, train_eval


def train_and_eval_with_generated_data(
    generated_data: Dict[int, List[Dict[str, str]]],
    evaluation_task: ClassificationEvalTask,
    num_epochs: int = 3,
    max_val_samples: int = 100,
    max_test_samples: int = 500,
):
    # format training data
    formatted_data = evaluation_task.format_for_training(generated_data)
    train_labels = [d[0] for d in formatted_data]
    train_texts = [d[1] for d in formatted_data]

    # Load real data for validation and testing
    val_texts, val_labels, test_texts, test_labels = load_val_test_data(
        evaluation_task, max_val_samples, max_test_samples
    )

    # Data Info
    core.logger.info("\nDataset sizes:")
    core.logger.info(f"Training: {len(train_texts)}")
    core.logger.info(f"Validation: {len(val_texts)}")
    core.logger.info(f"Test: {len(test_texts)}")

    # Train and eval
    train_data = (train_texts, train_labels)
    val_data = (val_texts, val_labels)
    test_data = (test_texts, test_labels)
    num_classes = evaluation_task.get_num_classes()
    metrics = train_eval(train_data, val_data, test_data, num_classes, num_epochs)

    core.logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    core.logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
    core.logger.info(f"Test Precision: {metrics['precision']:.4f}")
    core.logger.info(f"Test Recall: {metrics['recall']:.4f}")


def main_runner(
    source_pkl_file: str,
    evaluation_task: ClassificationEvalTask,
    num_epochs: int = 3,
    max_val_samples: int = 100,
    max_test_samples: int = 500,
):
    core.logger.info(f"Training and evaluating model for {source_pkl_file}")
    with open(source_pkl_file, "rb") as f:
        generated_data = pickle.load(f)
    train_and_eval_with_generated_data(
        generated_data, evaluation_task, num_epochs, max_val_samples, max_test_samples
    )


if __name__ == "__main__":
    # Importing example task
    from src.tasks import EnronClassificationTask as ExampleTask

    # classification evaluation example
    experiments = [
        {
            "source_pkl_file": "generated_data/enron/bare.pkl",
            "evaluation_task": ExampleTask(),
        },
    ]
    for i, parameter_set in enumerate(experiments):
        main_runner(**parameter_set)
        if i < len(experiments) - 1:
            core.reinitialize_logger()
