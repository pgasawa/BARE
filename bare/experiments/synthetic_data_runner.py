"""Runner for generating synthetic data via base-generation instruct-refinement"""

import copy
import time
import threading
import pickle
import os
import traceback

from pathlib import Path
from typing import List, Tuple, Callable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.logger as core
from src.tasks import DataGenerationTask, EvaluationTask
from experiments.generation_methods import GenerationMethod


def generation_wrapper(
    generation_fn: Callable[
        [DataGenerationTask, Dict[str, str]], tuple[List[Dict[str, str]], float]
    ],
    task: DataGenerationTask,
    grounding: Dict[str, list],
) -> Tuple[List[Dict[str, str]], float]:
    """Error-handling wrapper for data generation.

    Args:
        task: Task to generate data for
        generation_fn: Method to use for generation

    Returns:
        Tuple of (data_dict, total_cost)
    """
    try:
        core.logger.info("### BEGIN CALL OUTPUT ###")
        examples, cost = generation_fn(task, grounding)
        for example in examples:
            core.logger.info("###### BEGIN EXAMPLE ######")
            for k, v in example.items():
                if k != "context":
                    core.logger.info(f"{k}:\n{v}")
            core.logger.info("###### END EXAMPLE ######")
        core.logger.info("### END CALL OUTPUT ###\n")
        core.logger.flush_thread_log_queue(thread_id=threading.get_ident())
        return examples, cost

    except Exception as e:
        core.logger.error("".join(traceback.format_exception(e)))
        core.logger.flush_thread_log_queue(thread_id=threading.get_ident())

    return [], 0


def generate_data(
    num_calls: int,
    examples_per_call: int,
    generation_method: GenerationMethod,
    task: DataGenerationTask,
    source_data: List[Dict[str, str]],
    base_model: str,
    instruct_model: str,
) -> Tuple[List[Dict[str, str]], float]:
    """Generates the specified amount of synthetic data examples for the given
       domain using the specified method.

    Args:
        num_calls: Number of calls to make to generation method.
        examples_per_call: Number of examples to get per call.
        generation_method: Method to use for generation.
        task: Task to generate data for.
        source_data: Source data for refinement methods.
        base_model: Model to use for base generation methods.
        instruct_model: Model to use for instruction methods.
        analyze_diversity: Whether to analyze diversity of generated data.
        analyze_quality: Whether to analyze quality of generated data.

    Returns:
        Tuple of (List of data_dicts, total_cost)
    """
    core.logger.info(
        f"SYNTHETIC REFINEMENT - {generation_method.value} - {task.__class__.__qualname__} - n={num_calls}"
    )

    # validate parameters
    if task is None:
        raise ValueError("generation task cannot be None")
    if generation_method is None:
        raise ValueError("generation method cannot be None")
    if generation_method is GenerationMethod.REFINE_WITH_SOURCE:
        if num_calls != len(source_data):
            raise ValueError(
                f"mismatch between num_calls ({num_calls}) and source length ({len(source_data)})"
            )

    # start timing
    start_time = time.time()

    # get num_threads
    num_threads = min(90, os.cpu_count())

    # setup needed variables
    generated_data: List[Dict[str, str]] = []
    total_cost = 0.0
    inference_fn = generation_method.get_generation_function(
        examples_per_call=examples_per_call,
        source_data=source_data,
        base_model=base_model,
        instruct_model=instruct_model,
    )
    match generation_method:
        case (
            GenerationMethod.INSTRUCT_FEWSHOT_DYNAMIC
            | GenerationMethod.BASE_FEWSHOT_DYNAMIC
        ):
            task.build_fewshot_examples()

    # build grounding
    grounding = task.load_grounding(num_calls, seed=40)

    # add persona
    if generation_method == GenerationMethod.INSTRUCT_PERSONA:
        # obtain personas
        personas = task.load_personas()[0:examples_per_call]
        if len(personas) != examples_per_call:
            raise ValueError(
                f"Needed {examples_per_call} personas but only {len(personas)} provided"
            )

        # update inference function to single call
        inference_fn = generation_method.get_generation_function(
            examples_per_call=1,
            source_data=source_data,
            base_model=base_model,
            instruct_model=instruct_model,
        )

        # cartesian product of groundings and personas
        new_grounding = []
        for g in grounding:
            for p in personas:
                gg = copy.deepcopy(g)
                for k, v in p.items():
                    gg[f"persona.{k}"] = v
                new_grounding.append(gg)
        grounding = new_grounding
        assert len(grounding) == num_calls * examples_per_call
        num_calls = num_calls * examples_per_call
        examples_per_call = 1

    # submit generation jobs
    core.logger.set_multithreading()
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # submit jobs
            futures = [
                executor.submit(generation_wrapper, inference_fn, task, grounding[i])
                for i in range(num_calls)
            ]

            # process completions
            pbar = tqdm(total=num_calls, desc="Generating synthetic data")
            for future in as_completed(futures):
                generated_examples, cost = future.result()
                generated_data.extend(generated_examples)
                total_cost += cost
                pbar.update(1)
            pbar.close()

    # error logging
    except Exception as e:
        core.logger.error("".join(traceback.format_exception(e)))

    # end logger multithreading
    finally:
        core.logger.disable_multithreading()
        core.logger.flush_all_threads()

    core.logger.info(f"Final generation cost: ${total_cost:.4f}")
    core.logger.info(f"Latency: {time.time() - start_time}")
    return generated_data, total_cost


def generate_for_eval_task(
    num_calls: int,
    generation_method: GenerationMethod,
    domain: EvaluationTask,
    base_model: str,
    instruct_model: str,
    examples_per_call: int = 1,
    source_data: Dict[int, List[Dict[str, str]]] = None,
    save_location: str = None,
) -> Dict[int, List[Dict[str, str]]]:
    """Main function for generating synthetic data for a given task.

    Args:
        num_calls: Number of calls to make to generation method.
        generation_method: Method to use for generation.
        domain: Domain to generate data for.
        base_model: Model to use for base generation methods.
        instruct_model: Model to use for instruction methods.
        examples_per_call: Number of examples to generate from each call to generation_method.
        use_cached_source: Whether to use cached source data for refinement.
        source_data: Source data for refinement methods.
        save_location: Location to save the generated data.

    Returns:
       Dictionary of list of generated content by constituent task label.
    """
    core.logger.info(
        f"Generating synthetic training data for {domain.__class__.__qualname__}"
    )
    start_time = time.time()

    # verify parameters are ok
    if generation_method == GenerationMethod.REFINE_WITH_SOURCE:
        if source_data is None:
            raise ValueError(
                "base_source must be specified when using REFINE_WITH_SOURCE"
            )

    # setup tracked datastructures
    generated_data: Dict[int, List[Dict[str, str]]] = dict()
    generation_cost = 0

    # determine number of calls to make to each task
    generation_tasks = domain.constituent_tasks()
    num_calls_dict: Dict[int, int] = dict()
    if generation_method == GenerationMethod.REFINE_WITH_SOURCE:
        # match calls to instances for each task
        for task_id, _ in generation_tasks:
            if task_id not in source_data:
                num_calls_dict[task_id] = 0
            else:
                num_calls_dict[task_id] = len(source_data[task_id])
    else:
        # split calls evenly
        for idx, _ in generation_tasks:
            num_calls_dict[idx] = num_calls // len(generation_tasks)
            if idx < num_calls % len(generation_tasks):
                num_calls_dict[idx] += 1

    # perform generation for each task
    for task_id, generation_task in generation_tasks:
        # get number of calls
        if task_id not in num_calls_dict:
            domain_num_calls = 0
        else:
            domain_num_calls = num_calls_dict[task_id]

        # get source
        if generation_method == GenerationMethod.REFINE_WITH_SOURCE:
            source = source_data[task_id]
        else:
            source = None

        # generate
        domain_train_data, domain_generation_cost = generate_data(
            domain_num_calls,
            examples_per_call,
            generation_method,
            generation_task,
            source,
            base_model=base_model,
            instruct_model=instruct_model,
        )

        # add to tracked data
        generated_data[task_id] = domain_train_data
        generation_cost += domain_generation_cost

    core.logger.info(
        f"Generated {sum([len(d) for d in generated_data.values()])} synthetic examples"
    )
    core.logger.info(f"Generation Time: {time.time() - start_time:.2f}s")
    core.logger.info(f"Generation Cost: {generation_cost}")

    # save data
    if save_location is not None:
        savepath = Path(save_location)
        os.makedirs(savepath.parent.absolute(), exist_ok=True)
        with open(save_location, "wb") as f:
            pickle.dump(generated_data, f)

    return generated_data


if __name__ == "__main__":
    # Importing example task
    from src.tasks import EnronClassificationTask as ExampleTask

    # generation experiments example
    experiments = [
        {
            "num_calls": 500,
            "examples_per_call": 1,
            "generation_method": GenerationMethod.INSTRUCT_FEWSHOT,
            "domain": ExampleTask(),
            "source_data": None,
            "base_model": "vllm/llama-3.1-8b-base",
            "instruct_model": "gpt-4o",
            "save_location": "generated_data/enron/bare.pkl",
        },
    ]
    for i, parameter_set in enumerate(experiments):
        data = generate_for_eval_task(**parameter_set)
        if i < len(experiments) - 1:
            core.reinitialize_logger()
