"""Methods for synthetic data generation."""

import threading
import traceback

from enum import Enum
from typing import Callable, Dict, List

import src.logger as core
from src.tasks import DataGenerationTask
from src.generator.generator import Generator


instruct_temp = 1.2
base_temp = 0.7
refine_temp = 0.7


class GenerationMethod(Enum):
    """Enum class of synthetic refinement methods."""

    BASE_FEWSHOT = "base_fewshot"
    BASE_FEWSHOT_DYNAMIC = "base_fewshot_dynamic"
    INSTRUCT_FEWSHOT = "instruct_fewshot"
    INSTRUCT_FEWSHOT_DYNAMIC = "instruct_fewshot_dynamic"
    INSTRUCT_SEQUENTIAL = "instruct_sequential"
    INSTRUCT_IN_ONE = "instruct_in_one"
    INSTRUCT_PERSONA = "instruct_persona"
    BARE = "bare"
    REFINE_WITH_SOURCE = "refine_with_source"

    def get_generation_function(
        self,
        examples_per_call: int,
        source_data: List[Dict[str, str]],
        base_model: str,
        instruct_model: str,
    ) -> Callable[
        [DataGenerationTask, Dict[str, str]], tuple[List[Dict[str, str]], float]
    ]:
        """Obtains generation function corresponding to this enum.

        Args:
            examples_per_call: Number of examples to generate from each call to returned function.
            source_data: Source data for refinement methods.
            base_model: Model to use for base generation methods.
            instruct_model: Model to use for instruction methods.

        Returns:
            Generation function.
        """
        match self:
            case GenerationMethod.BASE_FEWSHOT:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: base_fewshot_static_generation(
                        task, grounding, base_model
                    ),
                )
            case GenerationMethod.BASE_FEWSHOT_DYNAMIC:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: base_fewshot_dynamic_generation(
                        task, grounding, base_model
                    ),
                )
            case GenerationMethod.INSTRUCT_FEWSHOT:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: instruct_fewshot_static_generation(
                        task, grounding, instruct_model
                    ),
                )
            case GenerationMethod.INSTRUCT_FEWSHOT_DYNAMIC:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: instruct_fewshot_dynamic_generation(
                        task, grounding, instruct_model
                    ),
                )
            case GenerationMethod.INSTRUCT_PERSONA:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: instruct_persona_generation(
                        task, grounding, instruct_model
                    ),
                )
            case GenerationMethod.BARE:
                return multicall_wrapper(
                    examples_per_call,
                    lambda task, grounding: refinement_generation(
                        task, grounding, base_model, instruct_model
                    ),
                )
            case GenerationMethod.REFINE_WITH_SOURCE:
                assert examples_per_call == 1, "only 1 per call supported"
                assert source_data is not None, "source must be specified"
                return refinement_generation_with_source(source_data, instruct_model)
            case GenerationMethod.INSTRUCT_SEQUENTIAL:
                return instruct_sequential(examples_per_call, instruct_model)
            case GenerationMethod.INSTRUCT_IN_ONE:
                return instruct_in_one(examples_per_call, instruct_model)
            case _:
                raise NotImplementedError


def multicall_wrapper(
    num: int,
    fn: Callable[
        [DataGenerationTask, Dict[str, str]], tuple[List[Dict[str, str]], float]
    ],
):
    """Wrapper for dynamically augmenting single-example generators into multi-example generators."""

    def multicall_fn(task: DataGenerationTask, grounding: Dict[str, str]):
        res = []
        cost = 0
        for _ in range(num):
            try:
                new_res, new_cost = fn(task, grounding)
                res.extend(new_res)
            except Exception as e:
                core.logger.error("".join(traceback.format_exception(e)))
                continue
            cost += new_cost
        return res, cost

    return multicall_fn


def base_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    dynamic: bool,
    stop: str,
    base_model: str,
):
    """Method for generating data from a base model.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        dynamic: Whether to use dynamic prompting
        stop: Stop sequence
        model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    generator = Generator(
        model_name=base_model,
        model_params={"temperature": base_temp, "stop": stop},
    )

    generator_response = generator.generate(
        prompt=task.base_prompt(dynamic=dynamic, grounding=grounding)
    )

    return [
        task.extract_content(generator_response.response, grounding=grounding)
    ], generator_response.cost


def base_fewshot_static_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    base_model: str,
):
    """Wrapper for generating from base model via few-shot prompting.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        base_model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    return base_generation(
        task=task,
        grounding=grounding,
        dynamic=False,
        stop=task.base_stop_tokens(),
        base_model=base_model,
    )


def base_fewshot_dynamic_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    base_model: str,
):
    """Wrapper for generating from base model via dynamic few-shot prompting.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        base_model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    return base_generation(
        task=task,
        grounding=grounding,
        dynamic=True,
        stop=task.base_stop_tokens(),
        base_model=base_model,
    )


def instruct_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    dynamic: bool,
    instruct_model: str,
):
    """Method for generating data from an instruct model.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        dynamic: Whether to use dynamic prompting
        model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    generator = Generator(
        model_name=instruct_model, model_params={"temperature": instruct_temp}
    )
    generator_response = generator.generate(
        prompt=task.instruct_prompt(grounding=grounding, dynamic=dynamic)
    )

    return [
        task.extract_content(generator_response.response, grounding=grounding),
    ], generator_response.cost


def instruct_fewshot_static_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    instruct_model: str,
):
    """Wrapper for generating from instruct model via few-shot prompting.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        instruct_model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    return instruct_generation(
        task,
        grounding=grounding,
        dynamic=False,
        instruct_model=instruct_model,
    )


def instruct_fewshot_dynamic_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    instruct_model: str,
):
    """Wrapper for generating from instruct model via dynamic few-shot prompting.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        instruct_model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    return instruct_generation(
        task,
        grounding=grounding,
        dynamic=True,
        instruct_model=instruct_model,
    )


def instruct_sequential(examples_per_call: int, instruct_model: str):
    """Function builder for generating sequentially from instruct model.

    Args:
        examples_per_call: The number of examples to generate in sequence.
        instruct_model: Model to use for generation

    Returns:
        Generation function.
    """

    def func(
        task: DataGenerationTask, grounding: Dict[str, str]
    ) -> tuple[List[Dict[str, str]], float]:
        """Method for generating sequentially from an instruct model.

        Args:
            task: Task to generate data for

        Returns: Tuple of (List[data_dict], total_cost)
        """
        generator = Generator(
            model_name=instruct_model, model_params={"temperature": instruct_temp}
        )

        # initial generation
        generator_response = generator.generate(
            prompt=task.instruct_prompt(grounding=grounding)
        )
        try:
            examples = [
                task.extract_content(generator_response.response, grounding=grounding)
            ]
        except Exception:
            print("Error in Extraction: {}".format(generator_response.response[0]))
            raise Exception
        cost = generator_response.cost

        # sequential generations
        for _ in range(examples_per_call - 1):
            generator_response = generator.generate(
                prompt=task.instruct_sequential_prompt(
                    grounding=grounding, prev_examples=examples
                )
            )
            try:
                examples.append(
                    task.extract_content(generator_response.response, grounding)
                )
            except Exception:
                core.logger.info(
                    "Error in Extraction: {}".format(generator_response.response[0])
                )
                continue
            cost += generator_response.cost

        return examples, cost

    return func


def instruct_in_one(examples_per_call: int, instruct_model: str):
    """Function builder for generating many examples at once from instruct model.

    Args:
        examples_per_call: The number of examples to generate at once.
        instruct_model: Model to use for generation

    Returns:
        Generation function.
    """

    def func(
        task: DataGenerationTask, grounding: Dict[str, str]
    ) -> tuple[List[Dict[str, str]], float]:
        """Method for generating many examples at once from an instruct model.

        Args:
            task: Task to generate data for

        Returns: Tuple of (List[data_dict], total_cost)
        """

        generator = Generator(
            model_name=instruct_model, model_params={"temperature": instruct_temp}
        )

        generator_response = generator.generate(
            prompt=task.instruct_in_one_prompt(
                grounding=grounding, num=examples_per_call
            )
        )
        raw_examples = task.split_instruct_in_one(generator_response.response)

        return [
            task.extract_content(d, grounding) for d in raw_examples
        ], generator_response.cost

    return func


def instruct_persona_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    instruct_model: str,
):
    """Method for generating data from an instruct model via persona prompting.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        instruct_model: Model to use for generation

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    generator = Generator(
        model_name=instruct_model, model_params={"temperature": instruct_temp}
    )
    generator_response = generator.generate(
        prompt=task.instruct_persona_prompt(grounding=grounding)
    )
    return [
        task.extract_content(generator_response.response, grounding=grounding),
    ], generator_response.cost


def refinement_generation(
    task: DataGenerationTask,
    grounding: Dict[str, str],
    base_model: str,
    instruct_model: str,
):
    """Method for generating data via instruct-refinement of base generations.

    Args:
        task: Task to generate data for
        grounding: Grounding context
        base_model: Model to use for base generation
        instruct_model: Model to use for instruction

    Returns:
        Tuple of ([data_dict], total_cost)
    """
    base_content, cost = base_generation(
        task,
        grounding=grounding,
        dynamic=False,
        stop=task.base_stop_tokens(),
        base_model=base_model,
    )
    for k, v in base_content[0].items():
        core.logger.info(f"base {k}:\n{v}")

    instruct_generator = Generator(
        model_name=instruct_model, model_params={"temperature": refine_temp}
    )
    instruct_response = instruct_generator.generate(
        prompt=task.instruct_refinement_prompt(base_content=base_content[0])
    )
    cost += instruct_response.cost
    # core.logger.info(f"instruct response:\n{instruct_response.response[0]}")
    return [task.extract_content(instruct_response.response, grounding)], cost


def refinement_generation_with_source(
    source: List[Dict[str, str]],
    instruct_model: str,
):
    """Method for generating data via instruct-refinement of provided base generations.

    Args:
        source: Source data to refine
        instruct_model: Model to use for instruction

    Returns:
        Generation function.
    """
    source_lock = threading.Lock()
    source_iter = iter(source)

    def func(task: DataGenerationTask, grounding: Dict[str, str]):
        instruct_generator = Generator(
            model_name=instruct_model, model_params={"temperature": refine_temp}
        )
        cost = 0.0
        with source_lock:
            base_content = next(source_iter)

        instruct_response = instruct_generator.generate(
            prompt=task.instruct_refinement_prompt(base_content=base_content)
        )
        cost += instruct_response.cost

        return [
            task.extract_content(
                instruct_response.response, task.separate_grounding(base_content)
            )
        ], cost

    return func
