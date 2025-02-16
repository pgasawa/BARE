"""Benchmark experiment against a dataset of PubMedQA problems."""

import time
import os
import threading
import string
import pandas as pd

from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.logger as core
from src.generator.generator import Generator
from src.tasks import DATASET_CACHE_DIR


def normalize_answer(s: str) -> str:

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return remove_punc(lower(s))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def process_problem(row: pd.DataFrame, model: str) -> tuple[bool, float]:
    """Takes in a row from the PubMedQA dataset, runs inference, and returns a tuple: (correctness, cost).

    Args:
        row: A row from the PubMedQA dataset.
        model: The model to use for inference.

    Returns:
        A tuple: (correctness, cost).
    """

    core.logger.info("##########")
    core.logger.info(f"Question ID: {row["question_id"]}")

    question = f"Question: {row["question"]}\nContext: {row["context"]}\n"
    question += "Think step by step then provide your yes/no/maybe answer at the end saying 'Answer:'."

    # Simple Inference
    generator = Generator(
        model_name=model,
        system_prompt="You are an AI assistant specializing in the medical domain tasked with answering research questions given multiple abstracts, some of which are relevant. Think step by step then answer at the end saying 'Answer: [yes, no, maybe]'.",
    )
    generator_response = generator.generate(prompt=question)
    try:
        extracted_answer = normalize_answer(
            generator_response.response.split("Answer: ")[1]
        )
    except Exception:
        core.logger.info(f"Error Parsing Response: {generator_response.response[0]}")
        core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
        return False, 0
    cost = generator_response.cost

    # Evaluation
    ground_truth_answer = row["answer"]
    correctness = exact_match_score(extracted_answer, ground_truth_answer)

    core.logger.info(
        f"""
Extracted Answer: {extracted_answer}
Ground Truth Answer: {ground_truth_answer}
Correctness: {correctness}"""
    )
    core.logger.info(f"ID Reference: {row["question_id"]}")
    core.logger.info(f"Full Reasoning: {generator_response.response[0]}")

    core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
    return correctness, cost


def main(num_samples: int, model: str, seed: int = None):
    global DATASET_CACHE_DIR

    start_time = time.time()
    num_threads = os.cpu_count() - 2
    core.logger.info(f"PubMedQA Experiment, over {num_samples} samples")
    if seed:
        core.logger.info(f"Random Seed: {seed}")

    dataset = load_from_disk(DATASET_CACHE_DIR + "/pub_med_qa_distractors")
    # Convert to dataframe with columns: id, question, answer, context

    df = pd.DataFrame()
    df["question_id"] = dataset["pubid"]
    df["question"] = dataset["question"]
    df["context"] = dataset["context"]
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
