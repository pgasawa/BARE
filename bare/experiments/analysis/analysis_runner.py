import pickle

import src.logger as core
from experiments.analysis.diversity_analysis_utils import analyze_diversity
from experiments.analysis.quality_analysis_utils import analyze_quality
from src.tasks import EvaluationTask


def analyze_data(
    source_pkl_file: str,
    domain: EvaluationTask,
    do_diversity_analysis: bool = True,
    do_quality_analysis: bool = True,
):
    core.logger.info(f"Analyzing diversity and quality for {source_pkl_file}")

    # open file
    with open(source_pkl_file, "rb") as f:
        generated_data = pickle.load(f)

    # get constituent tasks
    constituent_tasks = domain.constituent_tasks()

    # running sums
    total_cost = 0

    # separate analysis for different tasks
    for id, task in constituent_tasks:
        core.logger.info(f"Analysis for task {task.__class__.__qualname__}")
        task_data = generated_data[id]

        # diversity analysis
        if do_diversity_analysis:
            core.logger.info("Diversity:")
            avg_embedding_dist, cost = analyze_diversity(task, task_data)
            total_cost += cost
            core.logger.info(f"Avg. Embedding Distance: {avg_embedding_dist}")

        # quality analysis
        if do_quality_analysis:
            core.logger.info("Quality:")
            quality, cost = analyze_quality(task, task_data)
            total_cost += cost
            core.logger.info(f"Quality (IR): {quality}")

    core.logger.info(f"Total cost: {total_cost}")


if __name__ == "__main__":
    # Importing example task
    from src.tasks import EnronClassificationTask as ExampleTask

    # analysis example
    experiments = [
        {
            "source_pkl_file": "generated_data/enron/bare.pkl",
            "domain": ExampleTask(),
        },
    ]
    for i, parameter_set in enumerate(experiments):
        analyze_data(**parameter_set)
        if i < len(experiments) - 1:
            core.reinitialize_logger()
