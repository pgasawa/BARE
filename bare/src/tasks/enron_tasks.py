import random
from datasets import load_dataset
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, ClassificationEvalTask, DATASET_CACHE_DIR


class SpamDataGenerationTask(DataGenerationTask):
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
            # dynamic fewshot
            assert self.candidate_examples is not None
            examples_str = self.format_prompt_examples(
                random.sample(
                    self.candidate_examples,
                    min(len(self.candidate_examples), k),
                ),
                instruct=False,
            )
            return spam_base_prompt.format(
                examples=examples_str,
                n=min(len(self.candidate_examples), k) + 1,
            )
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                spam_static_examples[:k], instruct=False
            )
            return spam_base_prompt.format(
                examples=examples_str,
                n=len(spam_static_examples[:k]) + 1,
            )

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
            return spam_instruct_fewshot_prompt.format(examples=examples_str)
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                spam_static_examples[:k], instruct=True
            )
            return spam_instruct_fewshot_prompt.format(examples=examples_str)

    def instruct_sequential_prompt(
        self, grounding: Dict[str, str], prev_examples: List[Dict[str, str]]
    ) -> str:
        examples_str = self.format_prompt_examples(
            spam_static_examples + prev_examples, instruct=True
        )
        return spam_instruct_sequential_prompt.format(examples=examples_str)

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(spam_static_examples, instruct=True)
        return spam_instruct_in_one_prompt.format(num=num, examples=examples_str)

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EMAIL")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(spam_static_examples, instruct=True)
        return spam_instruct_persona_prompt.format(
            examples=examples_str, persona=grounding["persona.persona"]
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            spam_static_examples[:k], instruct=True
        )
        return spam_refinement_prompt.format(
            examples=examples_str, email=base_content["email"]
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            domain_text, domain_labels = EnronClassificationTask().load_train_data()
            self.candidate_examples = [
                {"email": example.strip()}
                for example, label in zip(domain_text, domain_labels)
                if label == 1
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [f"{d["email"]}" for d in data]
        else:
            formatted_strs = [
                f"Email {idx+1}:\n{d["email"]}\nEXAMPLE DONE"
                for idx, d in enumerate(data)
            ]
        return "\n\n".join(formatted_strs)

    def load_personas(self) -> List[Dict[str, str]]:
        return spam_personas

    # extraction methods
    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        return {"email": raw_response.strip()}

    # IR methods
    def get_quality_prompt(
        self,
        data_point: Dict[str, str],
        ground_truth_placement: int = 0,
        num_examples_to_choose_from: int = 4,
    ) -> str:
        self.build_fewshot_examples()
        examples = self.candidate_examples

        # Real examples
        random_samples = random.sample(examples, num_examples_to_choose_from - 1)

        # Build the questions in the right order based on ground_truth_placement
        questions_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                questions_compiled += f"Email {i}: {data_point['email']}\n"
            else:
                questions_compiled += (
                    f"Email {i}: {random_samples[real_example_idx]['email']}\n"
                )
                real_example_idx += 1

        return spam_ir_prompt.format(
            k=num_examples_to_choose_from,
            emails=questions_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return spam_ir_system_prompt.format(k=num_examples_to_choose_from)

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
        return generated_datum["email"]


class HamDataGenerationTask(DataGenerationTask):
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
            # dynamic fewshot
            assert self.candidate_examples is not None
            examples_str = self.format_prompt_examples(
                random.sample(
                    self.candidate_examples,
                    min(len(self.candidate_examples), k),
                ),
                instruct=False,
            )
            return ham_base_prompt.format(
                examples=examples_str,
                n=min(len(self.candidate_examples), k) + 1,
            )
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                ham_static_examples[:k], instruct=False
            )
            return ham_base_prompt.format(
                examples=examples_str,
                n=len(ham_static_examples[:k]) + 1,
            )

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
            return ham_instruct_fewshot_prompt.format(examples=examples_str)
        else:
            # static fewshot
            examples_str = self.format_prompt_examples(
                ham_static_examples[:k], instruct=True
            )
            return ham_instruct_fewshot_prompt.format(examples=examples_str)

    def instruct_sequential_prompt(
        self, grounding: Dict[str, str], prev_examples: List[Dict[str, str]]
    ) -> str:
        examples_str = self.format_prompt_examples(
            ham_static_examples + prev_examples, instruct=True
        )
        return ham_instruct_sequential_prompt.format(examples=examples_str)

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(ham_static_examples, instruct=True)
        return ham_instruct_in_one_prompt.format(num=num, examples=examples_str)

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EMAIL")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(ham_static_examples, instruct=True)
        return ham_instruct_persona_prompt.format(
            examples=examples_str, persona=grounding["persona.persona"]
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            ham_static_examples[:k], instruct=True
        )
        return ham_refinement_prompt.format(
            examples=examples_str, email=base_content["email"]
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            domain_text, domain_labels = EnronClassificationTask().load_train_data()
            self.candidate_examples = [
                {"email": example.strip()}
                for example, label in zip(domain_text, domain_labels)
                if label == 0
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [f"{d["email"]}" for d in data]
        else:
            formatted_strs = [
                f"Email {idx+1}:\n{d["email"]}\nEXAMPLE DONE"
                for idx, d in enumerate(data)
            ]
        return "\n\n".join(formatted_strs)

    def load_personas(self) -> List[Dict[str, str]]:
        return ham_personas

    # extraction methods
    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        return {"email": raw_response.strip()}

    # IR methods
    def get_quality_prompt(
        self,
        data_point: Dict[str, str],
        ground_truth_placement: int = 0,
        num_examples_to_choose_from: int = 4,
    ) -> str:
        self.build_fewshot_examples()
        examples = self.candidate_examples

        # Real examples
        random_samples = random.sample(examples, num_examples_to_choose_from - 1)

        # Build the questions in the right order based on ground_truth_placement
        questions_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                questions_compiled += f"Email {i}: {data_point['email']}\n"
            else:
                questions_compiled += (
                    f"Email {i}: {random_samples[real_example_idx]['email']}\n"
                )
                real_example_idx += 1

        return ham_ir_prompt.format(
            k=num_examples_to_choose_from,
            emails=questions_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return ham_ir_system_prompt.format(k=num_examples_to_choose_from)

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
        return generated_datum["email"]


class EnronClassificationTask(ClassificationEvalTask):
    def __init__(self):
        super().__init__()

    def get_num_classes(self) -> int:
        return 2

    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        """Obtain list of relevant generation tasks and associated ids.

        Returns:
            List of (id, Task) pairs
        """
        return [(0, HamDataGenerationTask()), (1, SpamDataGenerationTask())]

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        """Load training data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        dataset = load_dataset(
            "SetFit/enron_spam", split="train", cache_dir=DATASET_CACHE_DIR
        )
        self.train_data = (dataset["text"], dataset["label"])
        return self.train_data

    def load_val_data(self) -> Tuple[List[str], List[int]]:
        """Load validation data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        dataset = load_dataset(
            "SetFit/enron_spam", split="test", cache_dir=DATASET_CACHE_DIR
        )
        self.val_data = (dataset["text"][:1000], dataset["label"][:1000])
        return self.val_data

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        """Load test data for this task.

        Returns:
            Tuple of (List of texts, List of labels)
        """
        dataset = load_dataset(
            "SetFit/enron_spam", split="test", cache_dir=DATASET_CACHE_DIR
        )
        self.test_data = (dataset["text"][1000:], dataset["label"][1000:])
        return self.test_data

    def format_for_training(
        self, generated_data: Dict[int, List[Dict[str, str]]]
    ) -> List[Tuple[int, str]]:
        formatted_training_data = []
        for label, data in generated_data.items():
            formatted_training_data.extend([(label, d["email"]) for d in data])
        return [d for d in formatted_training_data if d[1] is not None]


# spam prompts

spam_personas = [
    {
        "persona": "You are a desparate freelancer who turned to writing scam emails for shady clients to make quick money. You are resourceful but ethically flexible, stressed by financial pressures, and willing to take risks for monetary gain."
    },
    {
        "persona": "You are a deceptive marketer, skilled in persuasive language, creating appealing messages, and understanding psychological triggers. You are persuasive, creative, pragmatic, and sees spamming as just another form of marketing."
    },
    {
        "persona": "You are a thrill-seeking adolescent who enjoys the attention of people you spam. You are reckless, curious, eager to prove themselves, and seeks validation from the spamming community."
    },
    {
        "persona": "You are a member of an organized scamming organization, following orders and benefiting from the security and resources offered by a large organization. Loyal to the organization, disciplined, cooperative, thrives in a structured environment with clear objectives."
    },
]

spam_static_examples = [
    {
        "email": """rolex watches starting under $ 199 . 99 buy your own rolex for only\n200 $\nand you - vip =\nperson\nyou won ' t believe your eyes with =\nthe quality and the expert craftmanship =\nour watches have .\nthe high profile jewelry\nyou ' re looking for at the lowest prices in the =\nnation !\nclick =\nhere and choose from our collection\nof rolex =\nwatches .", "rolex watches starting under $ 99 . 99 who can resist a 24 kt . white gold rolex watch surrounded in stainless steal ? the high profile jewelry you ' re looking for at the lowest prices in the nation ! click here and choose from our collection of rolex watches .\nremove"""
    },
    {
        "email": """our new greatt offr want to know how to save over 60 % o laudatory n your piils ?\nhttp : / / www . regis macartney teouse . com - successfull and proven w kimono ay to save your mon unattending ey .\ndreadnought v\na answerable g\ncockalorum al\nl lexical u\nwarship l\nbuffoonery rac blubbered l\ni feculent s macedonian val\nterzetto m\nandmanyother .\nbest prlces intuitive .\npericarditis high quaiity .\nworldwide photometry shlpplng .\ntotal conf rotter identiaiity .\n250 . 000 + satisfied custome deplenish rs .\ninaccurate have a nice day !"""
    },
    {
        "email": """software by the original manufacturer & at generic prices & software should be easy to use !\nall leaders strive to turn their followers into children .\ndance is the hidden language of the soul of the body .', 'your technology expert . software by the original manufacturer & at generic prices &\nhttp : / / iodlo . k 9 oh 5 j 2 dzu 2 rh 32 . goremd . biz\nwe are always in our own company .\nfor what is liberty but the unhampered translation of will into act ?"""
    },
]

spam_base_prompt = """Here are a few different examples of spam emails in the style of the Enron dataset. Note that each email is different.
{examples}

Email {n}
"""

spam_instruct_fewshot_prompt = """Provide an example of a spam email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lowercase and separate punctuation with spaces. Make sure to only provide the text of the email."""

spam_instruct_sequential_prompt = """Provide an example of a spam email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lower case and separate punctuation with spaces. Make sure to only provide the text of the email."""

spam_instruct_in_one_prompt = """Provide {num} examples of spam emails in the style of the Enron dataset. Provide only the text of the emails and nothing else except for delimiters. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lower case and separate punctuation with spaces. Delimit the end of an email with the phrase "END OF EMAIL" (all caps) on a new line. Otherwise, make sure to only provide the text of the email."""

spam_instruct_persona_prompt = """{persona}
Provide an example of a spam email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lowercase and separate punctuation with spaces. Make sure to only provide the text of the email."""

spam_refinement_prompt = """Improve the given spam email. Edit the email to be more coherent and similar in style to the Enron dataset. Do not change the theme of the email. Here are some examples of spam emails from the Enron dataset:
{examples}

Here is the email for you to edit:
{email}
Provide only the edited email in your response. Retain the same style as the Enron dataset: all lower case and separate punctuation with spaces. Also retain the theme of the original. Do not include any commentary or notes."""

spam_ir_system_prompt = """You are an expert at evaluating emails sent by spammers.

You will be shown {k} spam emails. All but one of them are generated from a high quality AI while one of them is of low quality.

Your task is to identify which email (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure and clarity
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each email carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

spam_ir_prompt = """Here are {k} spam emails. One of them is of low quality. Please identify which one:

{emails}

Analyze each email and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""


# ham prompts

ham_personas = [
    {
        "persona": "You are an entry-level employee, just starting out at a new company. You are eager to prove yourself and looking to stand out to your superiors."
    },
    {
        "persona": "You are a team leader, managing a small group working on important projects. You are a team player and eager to ensure everything runs smoothly."
    },
    {
        "persona": "You are a high-level executive, rising through the ranks over many years. You are loyal to your company and employees, and always looking for ways to improve your organization."
    },
    {
        "persona": "You are a senior employee, working at the company for many years but not as a manager. You are content with your position and are focused on finishing the work assigned to you and nothing more."
    },
]

ham_static_examples = [
    {
        "email": """funding deal louise , we have a customer that wants to monetize a couple of itm deals . cash out the door would be c $ 500 k with orig of c $ 300 k over ba ' s . please advise . thanks , rob"""
    },
    {
        "email": """fw : vince and ravi ,\nattached below is the resume of a student graduating from the university of\nchicago with an ms in mathematics . his specialty is financial mathematics\nand his coursework seems to have covered a lot of the sorts of things that i\nunderstand your group to do , vince . he is extremely interested in enron and\ncontacted me because i am an alum of the college at the u of c . i gather he\ndidn \' t go through enron \' s typical recruiting process for analysts or\nassociates because his background is more technical , than commercial . he\nacknowledged that his business experience is limited and from his resume you\ncan tell that he is young . he would therefore be willing to start as an\nanalyst or whatever the equivalent would be for a non - rotating position , but\nhis interest is in doing the kind of work that you do , vince , and ravi , i\nknow you have a similar background .\nplease let me know if this candidate would be of interest to either of you .\nif so , feel free to contact him directly . he would like to start work this\nsummer after graduating in june .\nthanks for your time and i hope this is useful to you .\nregards ,\nlaura\nlaura howenstine\nmanager , origination\nenron net works\n713 - 853 - 0308 ( office )\nlaura . howenstine @ enron . com\n- - - - - original message - - - - -\nfrom : " kodjo adovor " @ enron\n+ 40 enron @ enron . com ]\nsent : monday , february 26 , 2001 4 : 48 pm\nto : howenstine , laura\ncc : lhowenstine @ hotmail . com\nsubject :\ndear laura ,\nthanks for taking the time to talk to me about the opportunities available\nin financial derivatives at enron . i appreciate your help . my resume is\nattached to this email as resume . doc . i look forward to talking to you soon .\nregards ,\nn . kodjo adovor\nthe university of chicago\nfinancial mathematics\n- resume . doc"""
    },
    {
        "email": """fw : meeting with jeff skilling louise ,\nper our conversation of last week , you might be interested in the following meetings .\nk\n- - - - - original message - - - - -\nfrom : chapman , kay\nsent : wednesday , february 07 , 2001 5 : 55 pm\nto : taylor , liz ; heathman , karen ; daw , nicki ; taylor , liz ; kimberly hillis / hou / ect @ enron ; sera , sherri ; lehr , tonai ; watson , denys ; gutierrez , anabel\ncc : chapman , kay\nsubject : meeting with jeff skilling\ndave delainey has asked that i contact each of you for the following meetings :\ndate : february 22 , 2001 date : february 22 , 2001\nthursday thursday\ntime : 9 : 00 am - 9 : 45 am time : 9 : 45 am - 10 : 30 am\nlocation : mr . skilling ' s office location : mr . skilling ' s office\ntopic : charter review 2001 topic : charter review 2001\nattendees : jeff skilling attendees : jeff skilling\nrick buy rick buy\nmark frevert mark frevert\ndave delainey dave delainey\njohn lavorato john lavorato\njohn thompson michael l . miller\nscott josey\nif you have any questions , please feel free to call me .\nthanks ,\nkay 3 - 0643"""
    },
]

ham_base_prompt = """Here are a few different examples of legitimate emails in the style of the Enron dataset. Note that each email is different.
{examples}

Email {n}
"""

ham_instruct_fewshot_prompt = """Provide an example of a legitimate email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lowercase and separate punctuation with spaces. Make sure to only provide the text of the email."""

ham_instruct_sequential_prompt = """Provide an example of a legitimate email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lower case and separate punctuation with spaces. Make sure to only provide the text of the email."""

ham_instruct_in_one_prompt = """Provide {num} examples of legitimate emails in the style of the Enron dataset. Provide only the text of the emails and nothing else except for delimiters. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lower case and separate punctuation with spaces. Delimit the end of an email with the phrase "END OF EMAIL" (all caps) on a new line. Otherwise, make sure to only provide the text of the email."""

ham_instruct_persona_prompt = """{persona}
Provide an example of a legitimate email in the style of the Enron dataset. Provide only the text of the email and nothing else. Here are some examples:
{examples}

Now it's your turn. Your email should be different in content from the examples, but use the same style as the Enron dataset: all lowercase and separate punctuation with spaces. Make sure to only provide the text of the email."""

ham_refinement_prompt = """Improve the given legitimate email. Edit the email to be more coherent and similar in style to the Enron dataset. Do not change the theme of the email. Here are some examples of spam emails from the Enron dataset:
{examples}

Here is the email for you to edit:
{email}
Provide only the edited email in your response. Retain the same style as the Enron dataset: all lower case and separate punctuation with spaces. Also retain the theme of the original. Do not include any commentary or notes."""

ham_ir_system_prompt = """You are an expert at evaluating emails sent by employees at Enron.

You will be shown {k} emails. All but one of them are generated from a high quality AI while one of them is of low quality.

Your task is to identify which email (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure and clarity
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each email carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

ham_ir_prompt = """Here are {k} emails. One of them is of low quality. Please identify which one:

{emails}

Analyze each email and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
