import random
import json
from datasets import load_dataset
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, GenerativeEvalTask, DATASET_CACHE_DIR


class LCBDataGenerationTask(DataGenerationTask):
    def __init__(self):
        super().__init__()

    # base prompt related methods
    def base_stop_tokens(self) -> str:
        return "EXAMPLE DONE"

    def base_prompt(
        self,
        grounding: Dict[str, str] = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        if dynamic:
            assert self.candidate_examples is not None
            examples_str = self.format_prompt_examples(
                random.sample(
                    self.candidate_examples,
                    min(len(self.candidate_examples), k),
                ),
                instruct=False,
            )
            return lcb_base_prompt.format(examples=examples_str)
        else:
            examples_str = self.format_prompt_examples(
                lcb_static_examples[:k], instruct=False
            )
            return lcb_base_prompt.format(examples=examples_str)

    # instruct prompt related methods
    def instruct_prompt(
        self,
        grounding: str = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        if dynamic:
            assert self.candidate_examples is not None
            examples_str = self.format_prompt_examples(
                random.sample(
                    self.candidate_examples,
                    min(len(self.candidate_examples), k),
                ),
                instruct=True,
            )
            return lcb_instruct_fewshot_prompt.format(examples=examples_str)
        else:
            examples_str = self.format_prompt_examples(
                lcb_static_examples[:k], instruct=True
            )
            return lcb_instruct_fewshot_prompt.format(examples=examples_str)

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
        examples_str = self.format_prompt_examples(
            lcb_static_examples + prev_examples, instruct=True
        )
        return lcb_instruct_sequential_prompt.format(examples=examples_str)

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(lcb_static_examples, instruct=True)
        return lcb_instruct_in_one_prompt.format(num=num, examples=examples_str)

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EXAMPLE")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(lcb_static_examples, instruct=True)
        return lcb_persona_prompt.format(
            examples=examples_str, persona=grounding["persona.persona"]
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            lcb_static_examples[:k], instruct=True
        )
        return lcb_refine_prompt.format(
            examples=examples_str,
            question=base_content["question"],
            test_input=base_content["test_input"],
            reasoning=base_content["reasoning"],
            answer=base_content["answer"],
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            dataset = load_dataset(
                "livecodebench/test_generation",
                split="test",
                cache_dir=DATASET_CACHE_DIR,
            )

            self.candidate_examples = [
                {
                    "question": q,
                    "test_input": eval(t)[0]["input"],
                    "answer": eval(t)[0]["output"],
                }
                for q, t in zip(
                    dataset["question_content"],
                    dataset["test"],
                )
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [
                f"Question:\n{d['question']}\nTest Input:\n{d['test_input']}\nReasoning:\n{d['reasoning']}\nAnswer:\n{d['answer']}"
                for d in data
            ]
        else:
            formatted_strs = [
                "EXAMPLE START\n"
                + f"Question:\n{d['question']}\nTest Input:\n{d['test_input']}\nReasoning:\n{d['reasoning']}\nAnswer:\n{d['answer']}\n"
                + "EXAMPLE DONE"
                for d in data
            ]
        return "\n\n".join(formatted_strs)

    def load_personas(self) -> List[Dict[str, str]]:
        return lcb_personas

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        question_input_reasoning_answer = raw_response.split("Question:")[1]
        question = question_input_reasoning_answer.split("Test Input:")[0]
        test_input_reasoning_answer = question_input_reasoning_answer.split(
            "Test Input:"
        )[1]
        test_input = test_input_reasoning_answer.split("Reasoning:")[0]
        reasoning_answer = test_input_reasoning_answer.split("Reasoning:")[1]
        reasoning = reasoning_answer.split("Answer:")[0]
        answer = reasoning_answer.split("Answer:")[1]
        return {
            "question": question.strip(),
            "test_input": test_input.strip(),
            "reasoning": reasoning.strip(),
            "answer": answer.strip(),
        }

    # IR methods
    def get_quality_prompt(
        self,
        data_point: Dict[str, str],
        ground_truth_placement: int = 0,
        num_examples_to_choose_from: int = 4,
    ) -> str:
        self.build_fewshot_examples()
        examples = self.candidate_examples

        random_samples = random.sample(examples, num_examples_to_choose_from - 1)

        # Build the questions in the right order based on ground_truth_placement
        questions_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                questions_compiled += (
                    f"Question {i}: {data_point['question']}\n"
                    + "Test Input:\n"
                    + data_point["test_input"]
                    + "Answer:\n"
                    + data_point["answer"]
                )
            else:
                questions_compiled += (
                    f"Question {i}: {random_samples[real_example_idx]['question']}\n"
                    + "Test Input:\n"
                    + random_samples[real_example_idx]["test_input"]
                    + "Answer:\n"
                    + random_samples[real_example_idx]["answer"]
                )
                real_example_idx += 1

        return lcb_ir_prompt.format(
            k=num_examples_to_choose_from,
            questions=questions_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return lcb_ir_system_prompt.format(k=num_examples_to_choose_from)

    def parse_quality_response(
        self, response: str, ground_truth_placement: int = 0
    ) -> bool:
        # Extract the model's guess from the response
        lines = response.lower().split("\n")
        for line in reversed(lines):
            if "answer:" in line:
                # Extract the number after "answer:"
                try:
                    model_guess = int("".join(filter(str.isdigit, line)))
                    # Is acceptable if model guess is not the ground truth
                    return model_guess != ground_truth_placement
                except ValueError:
                    return False
        return False

    # diversity methods
    def format_for_embedding(self, generated_datum: Dict[str, str]) -> str:
        return (
            "Question:"
            + generated_datum["question"]
            + "\nTest Input:"
            + generated_datum["test_input"]
            + "\nReasoning:"
            + generated_datum["reasoning"]
            + "\nAnswer: "
            + generated_datum["answer"]
        )


class LCBGenerativeEvalTask(GenerativeEvalTask):
    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        return [(0, LCBDataGenerationTask())]

    def prep_for_training(
        self, generated_content: Dict[int, List[Dict[str, str]]], output_filename: str
    ):
        generated_content = generated_content[0]
        finetuning_data = []

        for i in range(len(generated_content)):
            try:
                question = "Question: {}\n".format(generated_content[i]["question"])
                question += "Test Input: {}\n".format(
                    generated_content[i]["test_input"]
                )
                question += "Think step by step, provide your reasoning after 'Reasoning:' then provide the answer at the end after the delimiter 'Answer:', like 'Answer: 24'."

                point = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant who is an expert at predicting the output of a function given its description and an input. Think step by step, provide your reasoning after 'Reasoning:' then provide the answer at the end after the delimiter 'Answer:', like 'Answer: 24'.",
                        },
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": "Reasoning: "
                            + generated_content[i]["reasoning"]
                            + "\nAnswer: "
                            + generated_content[i]["answer"],
                        },
                    ]
                }
                finetuning_data.append(point)
            except Exception:
                continue
        # Dump the finetuning data as a JSONL file
        with open(output_filename, "w") as f:
            for item in finetuning_data:
                f.write(f"{json.dumps(item)}\n")


lcb_static_examples = [
    {
        "question": "You are given a 0-indexed array of strings details. Each element of details provides information about a given passenger compressed into a string of length 15. The system is such that: The first ten characters consist of the phone number of passengers. The next character denotes the gender of the person. The following two characters are used to indicate the age of the person. The last two characters determine the seat allotted to that person. Return the number of passengers who are strictly more than 60 years old.",
        "test_input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]',
        "reasoning": "The age is the two characters after the gender. We see that for the first input, this is 75. For the second input, this is 92. For the third input, this is 40. We were asked how many passengers were strictly more than 60 years older. Therefore, there are two passengers above 60 years old.",
        "answer": "2",
    },
    {
        "question": "You are given a 0-indexed array nums of length n. The distinct difference array of nums is an array diff of length n such that diff[i] is equal to the number of distinct elements in the suffix nums[i + 1, ..., n - 1] subtracted from the number of distinct elements in the prefix nums[0, ..., i]. Return the distinct difference array of nums. Note that nums[i, ..., j] denotes the subarray of nums starting at index i and ending at index j inclusive. Particularly, if i > j then nums[i, ..., j] denotes an empty subarray.",
        "test_input": "[1, 2, 3, 4, 5]",
        "reasoning": "For the first element, we see there are four distinct elements (2,3,4,5) in the suffix versus one distinct element in the prefix (itself). Therefore, the distinct difference for the 0th element is 1 - 4 = -3. For the second element, we see there are three distinct elements (3,4,5) in the suffix and two distinct elements in the prefix (1,2). Therefore, the distinct difference for the 1st element is 2 - 3 = -1. For the third element, we see there are two distinct elements (4,5) in the suffix and three distinct elements in the prefix (1,2,3). Therefore, the distinct difference for the 2nd element is 3 - 2 = 1. For the fourth element, we see there is only one distinct element (5) in the suffix and four distinct elements in the prefix (1,2,3,4). Therefore, the distinct difference for the 3rd element is 4 - 1 = 3. Lastly, For the fifth element, we see there are no distinct elements in the suffix while there are five distinct elements in the prefix (1,2,3,4,5). Therefore, the distinct difference array is [-3, -1, 1, 3, 5].",
        "answer": "[-3, -1, 1, 3, 5]",
    },
    {
        "question": "You are given two integers, num and t. An integer x is called achievable if it can become equal to num after applying the following operation no more than t times: Increase or decrease x by 1, and simultaneously increase or decrease num by 1. Return the maximum possible achievable number. It can be proven that there exists at least one achievable number.",
        "test_input": "4\n1",
        "reasoning": "Since we are seeking to maximize x, we would always choose to increase our num (4) and decrease x by 1 at the same time. Since we are only allowed one operation, we can increase num to a max of 5. Since x can only be decreased once, the max achievable number x is 6.",
        "answer": "6",
    },
]


lcb_personas = [
    {
        "persona": "A meticulous computer science professor who values precision and clarity in problem statements."
    },
    {
        "persona": "A competitive programming enthusiast who enjoys writing tricky edge-case-heavy problems."
    },
    {
        "persona": "A playful educator who writes fun, story-driven questions to keep learners engaged."
    },
    {
        "persona": "A software engineer at a big tech company who focuses on real-world scenarios and efficiency concerns."
    },
    {
        "persona": "A mathematician who enjoys abstract, number-theoretic, or combinatorial problem formulations."
    },
    {
        "persona": "A cybersecurity expert who crafts algorithmic challenges inspired by encryption and security principles."
    },
    {
        "persona": "A technical recruiter who prioritizes practical, job-related coding problems for screening candidates."
    },
    {
        "persona": "An artificial intelligence researcher who incorporates machine learning-related themes in questions."
    },
    {
        "persona": "A historical scholar who frames algorithmic challenges around ancient computing methods and classical texts."
    },
    {
        "persona": "A game developer who designs problems based on game mechanics and simulation algorithms."
    },
]

lcb_persona_prompt = """You are {persona}
Provide an example of a natural language programming-esque task with a specified test input, reasoning, and the resulting output answer. Provide your example in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"
Provide only the question, test input, reasoning, and answer in the given format. Here are some examples:
{examples}

Now it's your turn. Start your response with the question, test input, reasoning, and answer."""

lcb_instruct_sequential_prompt = """Generate a new natural language programming-esque task with a specified test input, reasoning, and the resulting output answer. Here are the previously generated examples:
{examples}

Your new task should:
1. Be different from the previous examples
2. Follow the same format and style as prior tasks

Provide your new task in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"
Provide only the question, test input, reasoning, and answer in the given format. Start your response with the question."""

lcb_base_prompt = """Here are a few examples of natural language programming-esque tasks with a specified test input, reasoning, and the resulting output answer.

{examples}

EXAMPLE START
"""

lcb_instruct_fewshot_prompt = """Provide an example of a natural language programming-esque task with a specified test input, reasoning, and the resulting output answer. Provide your example in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"
Provide only the question, test input, reasoning, and answer in the given format. Here are some examples:
{examples}

Now it's your turn. Start your response with the question, test input, reasoning, and answer."""

lcb_instruct_sequential_prompt = """Generate a new natural language programming-esque task with a specified test input, reasoning, and the resulting output answer. Here are the previously generated examples:
{examples}

Your new task should:
1. Be different from the previous examples
2. Follow the same format and style as prior tasks

Provide your new task in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"
Provide only the question, test input, reasoning, and answer in the given format. Start your response with the question."""

lcb_instruct_in_one_prompt = """Provide {num} examples of natural language programming-esque tasks with a specified test input and the resulting output answer. Provide your examples in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"

Here are some examples:
{examples}

Now it's your turn. Generate {num} different problems following this format. Your question should be different in content from the examples. Make sure to only provide only the question, test input, reasoning, and answer. Start each example with the question. Delimit the end of an example with the phrase "END OF EXAMPLE" (all caps) on a new line."""

lcb_refine_prompt = """Improve the given natural language programming-esque task, test input, reasoning, and answer. Edit the task to be more similar in style to the examples. Do not change the theme of the task. Correct any errors and disambiguate as necessary. Provide your edited task in the following format:
"Question:
[question]
Test Input:
[test_input]
Reasoning:
[reasoning]
Answer:
[answer]"
Provide only the question, test input, reasoning, and answer in the given format. Here are some examples:
{examples}

Now it's your turn. Here is the question, test input, reasoning, and answer for you to edit:
Question:
{question}
Test Input:
{test_input}
Reasoning:
{reasoning}
Answer:
{answer}
Provide only the improved question, test input, reasoning, and answer in the given format. Do not include any commentary or notes. Start your response with the question."""

lcb_ir_system_prompt = """You are an expert at evaluating question, test input, answer pairs for natural langauge programming-esque output prediction tasks.

You will be shown {k} examples. Each examples consists of a question, a test input, and an answer. All but one of them is generated from a high quality AI while one of them is of low quality.

Your task is to identify which example (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure, clarity, and difficulty
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each example carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

lcb_ir_prompt = """Here are {k} examples. One of them is of low quality. Please identify which one:

{questions}

Analyze each example and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
