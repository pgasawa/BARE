"""Quality analysis for synthetic data using LLM evaluation."""

import os
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List

from src.generator.generator import GeneratorResponseObject, Generator
from src.tasks import DataGenerationTask


def _evaluate_single_point(
    task: DataGenerationTask,
    data_point: Dict[str, str],
    llm: str,
) -> tuple[bool, float]:
    """Evaluates a single data point.

    Args:
        task: The quality analysis task
        data_point: Data point to evaluate
        llm: LLM model name

    Returns:
        Tuple of (is_acceptable, cost)
    """
    num_examples_to_choose_from = 4

    quality_model = Generator(
        model_name=llm,
        system_prompt=task.get_quality_system_prompt(num_examples_to_choose_from),
    )

    # random number from 1 to k
    ground_truth_placement = random.randint(1, num_examples_to_choose_from)
    prompt = task.get_quality_prompt(
        data_point=data_point,
        ground_truth_placement=ground_truth_placement,
        num_examples_to_choose_from=num_examples_to_choose_from,
    )

    verifier_response: GeneratorResponseObject = quality_model.generate(prompt)
    is_acceptable = task.parse_quality_response(
        verifier_response.response,
        ground_truth_placement=ground_truth_placement,
    )
    return is_acceptable, verifier_response.cost


def analyze_quality(
    task: DataGenerationTask,
    data_points: List[Dict[str, str]],
    llm: str = "gpt-4o",
) -> tuple[float, float]:
    """Evaluates the quality of synthetic data points using LLM in parallel.

    Args:
        task: The type of quality analysis task
        data_points: List of synthetic data points to evaluate
        llm: Optional SimpleGenerator instance. If not provided, creates a new one.

    Returns:
        Float between 0 and 1 representing the fraction of acceptable quality data points
        Float representing the total cost
    """
    if not data_points:
        return 0.0, 0.0

    acceptable_count = 0
    total_cost = 0

    # Use ThreadPoolExecutor with number of CPUs
    num_threads = os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = [
            executor.submit(_evaluate_single_point, task, data_point, llm)
            for data_point in data_points
        ]

        # Process results as they complete
        with tqdm(total=len(data_points), desc="Evaluating Quality") as pbar:
            for future in as_completed(futures):
                is_acceptable, cost = future.result()
                if is_acceptable:
                    acceptable_count += 1
                total_cost += cost
                pbar.update(1)

    return acceptable_count / len(data_points), total_cost
