import random
from datasets import load_dataset
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, ClassificationEvalTask, DATASET_CACHE_DIR

# newsgroups classes
NEWSGROUPS_CLASSES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


class NewsgroupsDataGenerationTask(DataGenerationTask):
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
            # dynamic fewshot not implemented
            raise NotImplementedError
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                newsgroups_static_examples[:k], instruct=False
            )
            return newsgroups_base_prompt.format(examples=examples_str)

    # instruct prompt related methods
    def instruct_prompt(
        self,
        grounding: str = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        if dynamic:
            # dynamic fewshot
            assert self.candidate_examples is not None
            examples_str = self.format_prompt_examples(
                random.sample(
                    self.candidate_examples,
                    min(len(self.candidate_examples), k),
                ),
                instruct=True,
            )
            return newsgroups_instruct_fewshot_prompt.format(examples=examples_str)
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                newsgroups_static_examples[:k], instruct=True
            )
            return newsgroups_instruct_fewshot_prompt.format(examples=examples_str)

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
            newsgroups_static_examples + prev_examples, instruct=True
        )
        return newsgroups_instruct_sequential_prompt.format(examples=examples_str)

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(
            newsgroups_static_examples, instruct=True
        )
        return newsgroups_instruct_in_one_prompt.format(num=num, examples=examples_str)

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EXAMPLE")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(
            newsgroups_static_examples, instruct=True
        )
        return newsgroups_persona_prompt.format(
            examples=examples_str, persona=grounding["persona.persona"]
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            newsgroups_static_examples[:k], instruct=True
        )
        return newsgroups_refinement_prompt.format(
            examples=examples_str,
            newsgroup=base_content["newsgroup"],
            message=base_content["message"],
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            domain_text, domain_labels = (
                NewsgroupsClassificationEvalTask().load_train_data()
            )
            self.candidate_examples = [
                {"message": example.strip(), "newsgroup": NEWSGROUPS_CLASSES[label]}
                for example, label in zip(domain_text, domain_labels)
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [
                f"Newsgroup:\n{d["newsgroup"]}\nMessage:\n{d["message"]}" for d in data
            ]
        else:
            formatted_strs = [
                "EXAMPLE START\n"
                + f"Newsgroup:\n{d["newsgroup"]}\nMessage:\n{d["message"]}\n"
                + "EXAMPLE DONE"
                for d in data
            ]
        return "\n\n".join(formatted_strs)

    def load_personas(self) -> List[Dict[str, str]]:
        return newsgroups_personas

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        parsed = raw_response.split("Newsgroup:")[1].split("Message:")
        return {"newsgroup": parsed[0].strip(), "message": parsed[1].strip()}

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
        category = data_point["newsgroup"]
        if category not in NEWSGROUPS_CLASSES:
            return "Invalid"
        examples_same_category = [e for e in examples if e["newsgroup"] == category]
        random_sample_same_category = random.sample(
            examples_same_category, num_examples_to_choose_from - 1
        )

        # Build the questions in the right order based on ground_truth_placement
        questions_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                questions_compiled += f"Message {i}: {data_point['message']}\n"
            else:
                questions_compiled += f"Message {i}: {random_sample_same_category[real_example_idx]['message']}\n"
                real_example_idx += 1

        return newsgroups_ir_prompt.format(
            k=num_examples_to_choose_from,
            messages=questions_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return newsgroups_ir_system_prompt.format(k=num_examples_to_choose_from)

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
        return generated_datum["message"]


class NewsgroupsClassificationEvalTask(ClassificationEvalTask):
    """Interface for classification tasks to implement"""

    def __init__(self):
        super().__init__()

    def get_num_classes(self) -> int:
        return 20

    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        return [(0, NewsgroupsDataGenerationTask())]

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        dataset = load_dataset(
            "SetFit/20_newsgroups", split="train", cache_dir=DATASET_CACHE_DIR
        )
        self.train_data = (dataset["text"], dataset["label"])
        return self.train_data

    def load_val_data(self) -> Tuple[List[str], List[int]]:
        dataset = load_dataset(
            "SetFit/20_newsgroups", split="test", cache_dir=DATASET_CACHE_DIR
        )
        self.val_data = (dataset["text"][:1000], dataset["label"][:1000])
        return self.val_data

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        dataset = load_dataset(
            "SetFit/20_newsgroups", split="test", cache_dir=DATASET_CACHE_DIR
        )
        self.test_data = (dataset["text"][1000:], dataset["label"][1000:])
        return self.test_data

    def format_for_training(
        self, generated_data: Dict[int, List[Dict[str, str]]]
    ) -> List[Tuple[int, str]]:
        formatted_training_data = []
        generated_data = generated_data[0]
        for data in generated_data:
            try:
                label = NEWSGROUPS_CLASSES.index(data["newsgroup"])
            except ValueError:  # invalid class, discard example
                continue
            formatted_training_data.append((label, data["message"]))

        return [d for d in formatted_training_data if d[1] is not None]


newsgroups_personas = [
    {
        "persona": "You are a graduate student who recently discovered Usenet newsgroups. You are excited about the forum and are looking forward to sharing things you know."
    },
    {
        "persona": "You are an average company worker who uses Usenet newsgroups to discuss your interests. You are an enthusiast for all your hobbies and are very opinionated when it comes to them."
    },
    {
        "persona": "You are a senior employee who uses Usenet newsgroups to keep up-to-date in your field. You are an expert in your field and a curious individual who enjoys meaningful conversations with others."
    },
    {
        "persona": "You are a long-time user on Usenew newsgroups. You don't contribute very often but enjoy absorbing the information. Sometimes you contribute when there's a discussion that really catches your eye. You're an easy-going individual."
    },
]

newsgroups_static_examples = [
    {
        "newsgroup": """comp.sys.mac.hardware""",
        "message": """My wife has one of these.  I have not had much chance to fiddle with\nit, but in comparison to our Laserwriters with Canon engines, she\ncomplains that the print is too light for her taste.  The difference\nis quite apparent even when the print setting on the Select 310 is\nadjusted to the darkest possible level.  I don't find it\nobjectionable, and indeed rather like it, but be warned that some\npeople don't care for it and it is considerably different. """,
    },
    {
        "newsgroup": """talk.politics.mideast""",
        "message": """\nthe question is by going East or West from the misisipi. on either choice\nyou would loose Palestine or Broklyn, N.Y.\n\nI thought you\'re gonna say fromn misisipi back to the misisipi !\n\n\nLet\'s say : " let\'s establish the islamic state first" or "let\'s free our\noccupied lands first". And then we can dream about expansion, Mr. Gideon\n""",
    },
    {
        "newsgroup": """rec.motorcycles""",
        "message": """\nSuggest McQuires #1 plastic polish.  It will help somewhat but nothing \nwill remove deep scratches without making it worse than it already is.\nMcQuires will do something for fine or light stuff.\n\nAlso suggest calling your local plastic shop.  In Calif. "TAP PLASTIC" is\na chain that carries most of what is needed for repair and sometimes\nreplacement of plastic bits.  Telephone in the Bay area is 415-962-8430.\nI\'m not sure how amenable they are to shipping.  I have found that they\nhave several excellent products for cleaning, and removing crap from\nwindscreens and face shields.  Also they have one called "lift-it" which\nworks real well in removing sticky stuffs such as adhessives from plastic\nwihtout scratching same.\n\nLuck,""",
    },
]

newsgroups_base_prompt = """Here are a few examples of messages you might see from a newsgroup board. Note that each example comes from one of the following newsgroups: [comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, misc.forsale, talk.politics.misc, talk.politics.guns, talk.politics.mideast, talk.religion.misc, alt.atheism, soc.religion.christian]

{examples}

EXAMPLE START
"""

newsgroups_instruct_fewshot_prompt = """Provide an example of a message that might be found on a newsgroup message board, in the style of the 20 newsgroups dataset. Here are the list of newsgroups to choose from: [comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, misc.forsale, talk.politics.misc, talk.politics.guns, talk.politics.mideast, talk.religion.misc, alt.atheism, soc.religion.christian]. You must first specify the newsgroup your example is for, then provide the message. Provide your example in the following format:
"Newsgroup:
[newsgroup]
Message:
[message]"
Provide only the newsgroup and message in the given format. Here are some examples:
{examples}

Now it's your turn."""

newsgroups_instruct_sequential_prompt = """Provide an example of a message that might be found on a newsgroup message board, in the style of the 20 newsgroups dataset. Here are the list of newsgroups to choose from: [comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, misc.forsale, talk.politics.misc, talk.politics.guns, talk.politics.mideast, talk.religion.misc, alt.atheism, soc.religion.christian]. You must first specify the newsgroup your example is for, then provide the message. Provide your example in the following format:
"Newsgroup:
[newsgroup]
Message:
[message]"
Provide only the newsgroup and message in the given format. Here are some examples:
{examples}

Now it's your turn. Your message should be different in content from the examples. Make sure to only provide only the newsgroup and message."""

newsgroups_instruct_in_one_prompt = """Provide {num} examples of messages that might be found on a newsgroup message board, in the style of the 20 newsgroups dataset. Here are the list of newsgroups to choose from: [comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, misc.forsale, talk.politics.misc, talk.politics.guns, talk.politics.mideast, talk.religion.misc, alt.atheism, soc.religion.christian]. You must first specify the newsgroup your example is for, then provide the message. Provide your example in the following format:
"Newsgroup:
[newsgroup]
Message:
[message]"
Provide only the newsgroup and message in the given format. Here are some examples:
{examples}

Now it's your turn. Your message should be different in content from the examples. Make sure to only provide only the newsgroup and message. Delimit the end of an example with the phrase "END OF EXAMPLE" (all caps) on a new line. Otherwise, make sure to only provide the text of the email."""

newsgroups_persona_prompt = """{persona}
Provide an example of a message that might be found on a newsgroup message board, in the style of the 20 newsgroups dataset. Here are the list of newsgroups to choose from: [comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, misc.forsale, talk.politics.misc, talk.politics.guns, talk.politics.mideast, talk.religion.misc, alt.atheism, soc.religion.christian]. You must first specify the newsgroup your example is for, then provide the message. Provide your example in the following format:
"Newsgroup:
[newsgroup]
Message:
[message]"
Provide only the newsgroup and message in the given format. Here are some examples:
{examples}

Now it's your turn."""

newsgroups_refinement_prompt = """Improve the given message from the specified newsgroup message board, in the style of the 20 newsgroups dataset. Edit the message to bemore coherent and similar in style to the newsgroup dataset. Do not change the theme of the message. Provide your edited message in the following format, retaining the original newsgroup:
"Newsgroup:
[newsgroup]
Message:
[message]"
Provide only the newsgroup and message in the given format. Here are some examples of newsgroups and messages found in those newsgroups:
{examples}

Now it's your turn. Here is the newsgroup and the message for you to edit:
Newsgroup:
{newsgroup}
Message:
{message}
Provide only the original newsgroup and the improved message in the given format. Do not include any commentary or notes."""

newsgroups_ir_system_prompt = """You are an expert at evaluating messages sent on a Usenet newsgroup.

You will be shown {k} messages. All but one of them are generated from a high quality AI while one of them is of low quality.

Your task is to identify which message (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure and clarity
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each message carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

newsgroups_ir_prompt = """Here are {k} messages. One of them is of low quality. Please identify which one:

{messages}

Analyze each message and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
