"""Benchmark experiment against a dataset of GSM8K problems."""

import time
import os
import threading
import string
import pandas as pd

from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.logger as core
from src.generator.generator import Generator
from src.tasks import DATASET_CACHE_DIR


def strip_any_chars_except_numbers(s: str) -> str:
    return "".join(ch for ch in s if ch in string.digits)


def process_problem(row: pd.DataFrame, model: str) -> tuple[bool, float]:
    """Takes in a row from the GSM8K dataset, runs inference, and returns a tuple: (correctness, cost).

    Args:
        row: A row from the GSM8K dataset.
        model: The model to use for inference.

    Returns:
        A tuple: (correctness, cost).
    """

    core.logger.info("##########")

    question = f"Question: {row["question"]}\n"
    question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

    # Simple Inference
    generator = Generator(
        model_name=model,
        system_prompt="You are an AI assistant who is an expert at solving math word problems. Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'.",
        model_params={"temperature": 0.7},
    )
    generator_response = generator.generate(prompt=question)
    try:
        extracted_answer = generator_response.response.split("####")[1]
    except Exception:
        core.logger.info(f"Error Parsing Response: {generator_response.response[0]}")
        core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
        return False, 0
    cost = generator_response.cost

    # Evaluation
    ground_truth_answer = row["answer"]
    try:
        correctness = float(strip_any_chars_except_numbers(extracted_answer)) == float(
            strip_any_chars_except_numbers(ground_truth_answer.split("####")[1].strip())
        )
    except Exception:
        core.logger.info(
            f"Error Parsing Answer: {extracted_answer} or {ground_truth_answer}"
        )
        core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
        return False, 0

    core.logger.info(
        f"""
Extracted Answer: {extracted_answer}
Ground Truth Answer: {ground_truth_answer}
Correctness: {correctness}"""
    )
    core.logger.info(f"Full Reasoning: {generator_response.response[0]}")

    core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
    return correctness, cost


def main(num_samples: int, model: str, seed: int = None):
    global DATASET_CACHE_DIR

    start_time = time.time()
    num_threads = os.cpu_count()
    core.logger.info(f"GSM8K Experiment, over {num_samples} samples")
    if seed:
        core.logger.info(f"Random Seed: {seed}")

    dataset = load_dataset(
        path="openai/gsm8k",
        name="main",
        split="test",
        cache_dir=DATASET_CACHE_DIR,
    )
    # Convert to dataframe with columns: id, question, answer, context

    df = pd.DataFrame()
    df["question"] = dataset["question"]
    df["answer"] = dataset["answer"]

    total_cost = 0
    correctness = []

    try:
        core.logger.set_multithreading()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_problem, row, model)
                for _, row in df.sample(num_samples, random_state=seed).iterrows()
            ]

            with tqdm(
                total=num_samples, desc="Processing Problems", leave=True
            ) as pbar:
                for future in as_completed(futures):
                    result_correctness, result_cost = future.result()
                    correctness.append(result_correctness)
                    total_cost += result_cost
                    pbar.update(1)

        core.logger.disable_multithreading()

    except Exception as e:
        core.logger.disable_multithreading()
        core.logger.flush_all_threads()
        core.logger.error(f"Error: {e}")

    core.logger.info(f"Num Samples: {num_samples}")
    core.logger.info(f"Accuracy: {sum(correctness) / len(correctness)}")
    core.logger.info(f"Total Cost: ${total_cost}")
    core.logger.info(f"Latency: {time.time() - start_time}")


if __name__ == "__main__":
    # Example usage
    main(num_samples=100, model="gpt-4o-mini", seed=40)
