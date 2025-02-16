# This file contains base task classes for all domains.
from typing import Dict, List, Tuple

# dataset location
DATASET_CACHE_DIR = "./local_dataset_cache"


class DataGenerationTask:
    """Interface for all generation tasks to implement."""

    def __init__(self):
        """Initialize the task. Sets candidate_examples to None."""
        self.candidate_examples: List[Dict[str, str]] | None = None

    # base prompt related methods
    def base_stop_tokens(self) -> str:
        """Returns the stop tokens for base prompts.

        Returns:
            `str` of stop tokens.
        """
        raise NotImplementedError

    def base_prompt(
        self,
        grounding: Dict[str, str] = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        """Returns base prompt.

        Args:
            grounding: dictionary of grounding to use.
            dynamic: `True` if dynamic fewshot prompting, `False` otherwise.
            k: maximum number of examples to use.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    # instruct prompt related methods
    def instruct_prompt(
        self,
        grounding: str = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        """Returns instruct prompt.

        Args:
            grounding: dictionary of grounding to use.
            dynamic: `True` if dynamic fewshot prompting, `False` otherwise.
            k: maximum number of examples to use.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def instruct_sequential_prompt(
        self, grounding: Dict[str, str], prev_examples: List[Dict[str, str]]
    ) -> str:
        """Returns instruct prompt for sequential generation.

        Args:
            grounding: dictionary of grounding to use.
            prev_examples: List of extracted outputs from previous generations.

        Returns:
            The new prompt.
        """
        raise NotImplementedError

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        """Returns instruct prompt for generating multiple examples at once.

        Args:
            grounding: dictionary of grounding to use.
            num: Number of examples to obtain.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        """Splits raw model output for instruct in-one calls to separate examples.

        Args:
            raw_response: Text of raw model output.

        Returns:
            List of `str`s for each generated example.
        """
        raise NotImplementedError

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        """Returns instruct prompt using personas.

        Args:
            grounding: dictionary of grounding to use, including persona.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        """Returns instruct prompt for refining existing examples.

        Args:
            base_content: Extracted content of example to refine.
            k: Number of examples provided.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    # general prompt methods
    def build_fewshot_examples(self):
        """Loads candidate_examples if not already loaded."""
        raise NotImplementedError

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        """Formats list of examples for inclusion in prompts.

        Args:
            data: List of data dicts of examples.
            instruct: true if for use in instruct models, false if for use in base models.

        Returns:
            Formatted string for inclusion in prompts.
        """
        raise NotImplementedError

    def load_grounding(self, num: int, seed: int) -> List[Dict[str, str]]:
        """Returns grounding to be used during generation.

        Args:
            num: number of grounding dictionaries to obtain.
            seed: random seed.

        Returns:
            List of grounding dictionaries.
        """
        return [dict() for _ in range(num)]

    def load_personas(self) -> List[Dict[str, str]]:
        """Loads personas to use for persona prompting.

        Returns:
            List of persona dictionaries.
        """
        raise NotImplementedError

    # extraction methods
    def separate_grounding(self, content: Dict[str, str]) -> Dict[str, str]:
        """Extracts formatted grounding from output. For use when refining from source.

        Args:
            content: parsed model response.

        Returns:
            Dictionary with just formatted grounding.
        """
        return {"context": ""}

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        """Extracts content from raw model output. Includes grounding.

        Args:
            raw_response: Text of raw model output.
            grounding: grounding used.

        Returns:
            `Dict` with extracted content and formatted grounding.
        """
        raise NotImplementedError

    # IR methods
    def get_quality_prompt(
        self,
        data_point: Dict[str, str],
        ground_truth_placement: int = 0,
        num_examples_to_choose_from: int = 4,
    ) -> str:
        """Gets the IR evaluation prompt.

        Args:
            data_point: dictionary of parsed model response.
            ground_truth_placement: index to place model generation.
            num_examples_to_choose_from: total number of examples in prompt.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        """Gets the IR evaluation system prompt.

        Args:
            num_examples_to_choose_from: total number of examples per prompt.

        Returns:
            The system prompt.
        """
        raise NotImplementedError

    def parse_quality_response(
        self, response: str, ground_truth_placement: int = 0
    ) -> bool:
        """Parses IR evaluation result.

        Args:
            response: raw response from model.
            ground_truth_placement: location of ground truth.

        Returns:
            Correctness of response.
        """
        raise NotImplementedError

    # diversity methods
    def format_for_embedding(self, generated_datum: Dict[str, str]) -> str:
        """Formats single data point into a string for embedding calculation purposes.

        Args:
            generated_datum: generated data point.

        Returns:
            String representation.
        """
        raise NotImplementedError


class EvaluationTask:
    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        """Obtain list of relevant generation tasks and associated ids.

        Returns:
            List of (id, Task) pairs
        """
        raise NotImplementedError


class ClassificationEvalTask(EvaluationTask):
    """Interface for classification tasks to implement"""

    def __init__(self):
        """Initialize the task. Sets all data fields to `None`."""
        super().__init__()
        self.train_data: Tuple[List[str], List[int]] | None = None
        self.val_data: Tuple[List[str], List[int]] | None = None
        self.test_data: Tuple[List[str], List[int]] | None = None

    def get_num_classes(self) -> int:
        """Obtain number of classes."""
        raise NotImplementedError

    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        """Obtain list of relevant generation tasks and associated ids.

        Returns:
            List of (id, Task) pairs
        """
        raise NotImplementedError

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        """Load training data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        raise NotImplementedError

    def load_val_data(self) -> Tuple[List[str], List[int]]:
        """Load validation data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        raise NotImplementedError

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        """Load test data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        raise NotImplementedError

    def format_for_training(
        self, generated_data: Dict[int, List[Dict[str, str]]]
    ) -> List[Tuple[int, str]]:
        """Formats generated data for training a classification model.

        Args:
            generated_data: data generated.

        Returns:
            List of Tuples of (label, text).
        """


class GenerativeEvalTask(EvaluationTask):
    """Interface for generative tasks to implement"""

    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        """Obtain list of relevant generation tasks and associated ids.

        Returns:
            List of (id, Task) pairs
        """
        raise NotImplementedError

    def prep_for_training(
        self, generated_content: Dict[int, List[Dict[str, str]]], output_filename: str
    ):
        """Prepares parsed model generations for training.

        Args:
            generated_content: parsed model generations.
            output_filename: location to store formatted training data.
        """
        raise NotImplementedError
