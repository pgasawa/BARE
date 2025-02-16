import random
import json
from datasets import load_dataset
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, GenerativeEvalTask, DATASET_CACHE_DIR


class GSM8KDataGenerationTask(DataGenerationTask):
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
            return gsm8k_base_prompt.format(examples=examples_str)
        else:
            examples_str = self.format_prompt_examples(
                gsm8k_static_examples[:k], instruct=False
            )
            return gsm8k_base_prompt.format(examples=examples_str)

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
            return gsm8k_instruct_fewshot_prompt.format(examples=examples_str)
        else:
            examples_str = self.format_prompt_examples(
                gsm8k_static_examples[:k], instruct=True
            )
            return gsm8k_instruct_fewshot_prompt.format(examples=examples_str)

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
            gsm8k_static_examples + prev_examples, instruct=True
        )
        return gsm8k_instruct_sequential_prompt.format(examples=examples_str)

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(gsm8k_static_examples, instruct=True)
        return gsm8k_instruct_in_one_prompt.format(num=num, examples=examples_str)

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EXAMPLE")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(gsm8k_static_examples, instruct=True)
        return gsm8k_persona_prompt.format(
            examples=examples_str, persona=grounding["persona.persona"]
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            gsm8k_static_examples[:k], instruct=True
        )
        return gsm8k_refine_prompt.format(
            examples=examples_str,
            question=base_content["question"],
            answer=base_content["answer"],
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            dataset = load_dataset(
                "openai/gsm8k",
                name="main",
                split="train",
                cache_dir=DATASET_CACHE_DIR,
            )
            self.candidate_examples = [
                {"question": q, "answer": a}
                for q, a in zip(dataset["question"], dataset["answer"])
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [
                f"Question:\n{d['question']}\nAnswer:\n{d['answer']}" for d in data
            ]
        else:
            formatted_strs = [
                "EXAMPLE START\n"
                + f"Question:\n{d['question']}\nAnswer:\n{d['answer']}\n"
                + "EXAMPLE DONE"
                for d in data
            ]
        return "\n\n".join(formatted_strs)

    def load_personas(self) -> List[Dict[str, str]]:
        return gsm8k_personas

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        parsed = raw_response.split("Question:")[1].split("Answer:")
        return {
            "question": parsed[0].strip(),
            "answer": parsed[1].strip(),
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

        # Examples from same category as data point
        random_samples = random.sample(examples, num_examples_to_choose_from - 1)

        # Build the questions in the right order based on ground_truth_placement
        questions_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                questions_compiled += f"Question {i}: {data_point['question']}\n + Answer: {data_point['answer']}\n"
            else:
                questions_compiled += f"Question {i}: {random_samples[real_example_idx]['question']}\n + Answer: {random_samples[real_example_idx]['answer']}\n"
                real_example_idx += 1

        return gsm8k_ir_prompt.format(
            k=num_examples_to_choose_from,
            questions=questions_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return gsm8k_ir_system_prompt.format(k=num_examples_to_choose_from)

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
        return generated_datum["question"] + "\nAnswer: " + generated_datum["answer"]


class GSM8KGenerativeEvalTask(GenerativeEvalTask):
    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        return [(0, GSM8KDataGenerationTask())]

    def prep_for_training(
        self, generated_content: Dict[int, List[Dict[str, str]]], output_filename: str
    ):
        generated_content = generated_content[0]
        finetuning_data = []

        for i in range(len(generated_content)):
            try:
                question = "Question: {}\n".format(generated_content[i]["question"])
                question += "Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'."

                point = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant who is an expert at solving math word problems. Think step by step then provide the numerical answer at the end after the delimiter '####', like '#### 24'.",
                        },
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": "Answer: " + generated_content[i]["answer"],
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


gsm8k_personas = [
    {
        "persona": "a whimsical storyteller who weaves word problems into fairy tales and adventure stories."
    },
    {
        "persona": "a math-loving detective who creates mystery-themed problems where students solve for missing clues."
    },
    {
        "persona": "a sports analyst who frames problems around game statistics, scores, and player performance."
    },
    {
        "persona": "a futuristic scientist who designs word problems based on space travel, robots, and time machines."
    },
    {
        "persona": "a chef and restaurateur who poses problems involving recipes, ingredient measurements, and restaurant management."
    },
    {
        "persona": "a video game designer who builds problems around game mechanics, in-game currencies, and player strategies."
    },
    {
        "persona": "a zookeeper who writes problems involving animal populations, feeding schedules, and habitat design."
    },
    {
        "persona": "an environmentalist who frames problems around resource conservation, recycling, and energy use."
    },
    {
        "persona": "a treasure hunter who creates problems involving maps, coordinates, and hidden treasures."
    },
    {
        "persona": "a travel blogger who writes about distances, currency exchange, and trip planning in different countries."
    },
]

gsm8k_static_examples = [
    {
        "question": """Alice has 20 quarters. She wants to exchange them for nickels and so she goes to the bank. After getting back from the bank, she discovers that 20% of the nickels are iron nickels worth $3 each. What is the total value of her money now?""",
        "answer": "A quarter is worth five nickels because .25 / .05 = <<.25/.05=5>>5 She gets 100 nickels from the bank because 20 x 5 = <<20*5=100>>100 20 of the nickels are iron nickels because 100 x .20 = <<100*.20=20>>20 80 of the nickels are regular because 100 - 20 = <<100-20=80>>80 The iron nickels are worth $60 because 20 x 3 = <<20*3=60>>60 The regular nickels are worth $4 because 80 x .05 = <<80*.05=4>>4 Her money is now worth $64 because 60 + 4 = <<60+4=64>>64 #### 64",
    },
    {
        "question": "A church has 120 members. 40% are adults. The rest are children. How many children more children are there than adults?",
        "answer": "There are 48 adults because 120 x .4 = <<120*.4=48>>48 60% of members are children because 100 - 40 = <<100-40=60>>60 There are 72 children because 120 x .6 = <<120*.6=72>>72 There are 24 more children than adults because 72 - 48 = <<72-48=24>>24 #### 24",
    },
    {
        "question": "Lisa is looking to attempt a World Record. She has decided to try and match Joey Chestnut's record of eating 75 full hotdogs, buns included, in 10 minutes. Halfway through the time Lisa has eaten 20 hotdogs. How many hotdogs will she have to eat per minute to at least tie Joey Chestnut's record?",
        "answer": "Joey Chestnut ate 75 hotdogs to claim the record and Lisa has eaten 20 hot dogs so far, so she still needs to eat 75-20=<<75-20=55>>55 hotdogs to tie Joey Chestnut. Lisa has a 10 minute time period to eat the hotdogs and half the time has already passed, which means Lisa has 10/2=<<10/2=5>>5 minutes left until the competition is over. If she needs to eat 55 hotdogs to tie Joey Chestnut and there are 5 minutes left in the competition period then she needs to eat 55/5=<<55/5=11>>11 hot dogs per minute to have a chance of tying for a win. #### 11",
    },
]

gsm8k_base_prompt = """Here are a few examples of grade school math word problems that require performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. The numerical answer is provided at the end of each example after ####.

{examples}

EXAMPLE START
"""

gsm8k_instruct_fewshot_prompt = """Provide an example of a grade school math word problem that requires performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. You must first specify the question, then provide the very concise reasoning and answer. Provide your example in the following format:
"Question:
[question]
Answer:
[answer]"
Provide only the question and answer in the given format. Note how the numerical answer is provided after #### after each brief reasoning for a question. Here are some examples:
{examples}

Now it's your turn. Start your response with the question."""

gsm8k_persona_prompt = """You are {persona}
Provide an example of a grade school math word problem that requires performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. You must first specify the question, then provide the very concise reasoning and answer. Provide your example in the following format:
"Question:
[question]
Answer:
[answer]"
Provide only the question and answer in the given format. Note how the numerical answer is provided after #### after each brief reasoning for a question. Here are some examples:
{examples}

Now it's your turn. Start your response with the question."""

gsm8k_instruct_sequential_prompt = """Generate a new grade school math word problem that requires performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. Here are the previously generated examples:
{examples}

Your new problem should:
1. Be different from the previous examples
2. Follow the same format and style as prior problems

Note how the numerical answer is provided after #### after each brief reasoning for a question. Provide only the question and answer in the given format here:
"Question:
[question]
Answer:
[answer]" Start your response with the question."""

gsm8k_instruct_in_one_prompt = """Provide {num} examples of problems that might be grade school math word problem that requires performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra.You must first specify the question then provide the brief reasoning and answer. Note how the numerical answer is provided after #### after each brief reasoning for a question. Provide your examples in the following format:
"Question:
[question]
Answer: [answer]"

Here are some examples:
{examples}

Now it's your turn. Generate {num} different problems following this format. Your question should be different in content from the examples. Make sure to only provide only the question and answer.Start each example with the question. Delimit the end of an example with the phrase "END OF EXAMPLE" (all caps) on a new line."""

gsm8k_refine_prompt = """Improve the given grade school math word problem. Edit the problem or answer to be more similar in style to the examples, and disambiguate as necessary, in addition to correcting any errors. Do not change the theme of the problem. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. Note how the numerical answer is provided after #### after each brief reasoning for a question. Provide your edited problem in the following format:
"Question:
[question]
Answer: [answer]"
Provide only the question and answer in the given format. Here are some examples of categories and problems on those categories:
{examples}

Now it's your turn. Here is the question and anwer for you to edit:
Question:
{question}
Answer:
{answer}
Provide only the improved question and answer in the given format. Do not include any commentary or notes. Start your response with the question."""

gsm8k_ir_system_prompt = """You are an expert at evaluating question and answer pairs for grade school math word problems.

You will be shown {k} examples. Each examples consists of some context, a question, and an answer. All but one of them is generated from a high quality AI while one of them is of low quality.

Your task is to identify which example (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure, clarity, and difficulty
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each example carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

gsm8k_ir_prompt = """Here are {k} examples. One of them is of low quality. Please identify which one:

{questions}

Analyze each example and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
