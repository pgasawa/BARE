import random
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, GenerativeEvalTask, DATASET_CACHE_DIR


class HotpotQADataGenerationTask(DataGenerationTask):
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
                hotpotqa_static_examples[:k], instruct=False
            )
            return hotpotqa_base_prompt.format(
                examples=examples_str, context=grounding["context"]
            )

    # instruct prompt related methods
    def instruct_prompt(
        self,
        grounding: str = None,
        dynamic: bool = False,
        k: int = 3,
    ) -> str:
        if dynamic:
            # dynamic fewshot not implemented
            raise NotImplementedError
        else:
            examples_str = self.format_prompt_examples(
                hotpotqa_static_examples[:k], instruct=True
            )
            return hotpotqa_instruct_fewshot_prompt.format(
                examples=examples_str, context=grounding["context"]
            )

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
            hotpotqa_static_examples + prev_examples, instruct=True
        )
        return hotpotqa_instruct_sequential_prompt.format(
            examples=examples_str, context=grounding["context"]
        )

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(
            hotpotqa_static_examples, instruct=True
        )
        return hotpotqa_instruct_in_one_prompt.format(
            num=num, examples=examples_str, context=grounding["context"]
        )

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EXAMPLE")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(
            hotpotqa_static_examples, instruct=True
        )
        return hotpotqa_persona_prompt.format(
            examples=examples_str,
            persona=grounding["persona.persona"],
            context=grounding["context"],
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            hotpotqa_static_examples[:k], instruct=True
        )
        return hotpotqa_refine_prompt.format(
            examples=examples_str,
            context=base_content["context"],
            question=base_content["question"],
            reason=base_content["reason"],
            answer=base_content["answer"],
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            dataset = load_dataset(
                "hotpotqa/hotpot_qa",
                "distractor",
                split="validation",
                cache_dir=DATASET_CACHE_DIR,
            )
            formatted_context = [
                f"{"\n".join([s for ss in g["sentences"] for s in ss])}\n"
                for g in dataset["context"]
            ]
            self.candidate_examples = [
                {
                    "context": c,
                    "question": q,
                    "answer": a,
                }
                for c, q, a in zip(
                    formatted_context, dataset["question"], dataset["answer"]
                )
            ]

    def format_prompt_examples(self, data: List[Dict[str, str]], instruct: bool) -> str:
        if instruct:
            formatted_strs = [
                f"##Context:\n{d["context"]}\n"
                + f"##Question: {d["question"]}\n"
                + f"##Reason: {d["reason"]} "
                + f"##Answer: {d["answer"]}"
                for d in data
            ]
        else:
            formatted_strs = [
                "EXAMPLE START\n"
                + f"##Context:\n{d["context"]}\n"
                + f"##Question: {d["question"]}\n"
                + f"##Reason: {d["reason"]} "
                + f"##Answer: {d["answer"]}\n"
                + "EXAMPLE DONE"
                for d in data
            ]
        return "\n\n".join(formatted_strs)

    def load_grounding(self, num: int, seed: int) -> List[Dict[str, str]]:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
            cache_dir=DATASET_CACHE_DIR,
        )
        df = pd.DataFrame()
        df["context"] = dataset["context"]
        df = df.sample(n=num, random_state=seed)
        grounding = df["context"].tolist()
        formatted_grounding = [
            {"context": f"{"\n".join([s for ss in g["sentences"] for s in ss])}\n"}
            for g in grounding
        ]
        return formatted_grounding

    def load_personas(self) -> List[Dict[str, str]]:
        return hotpotqa_personas

    def separate_grounding(self, content):
        return {"context": content["context"]}

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        question_reason_answer = raw_response.split("##Question:")[1]
        question = question_reason_answer.split("##Reason:")[0]
        reason_answer = question_reason_answer.split("##Reason:")[1]
        reason, answer = (
            reason_answer.split("##Answer:")[0],
            reason_answer.split("##Answer:")[1],
        )

        return {
            "question": question.strip(),
            "reason": reason.strip(),
            "answer": answer.strip(),
            "context": grounding["context"],
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
        random_sample = random.sample(examples, num_examples_to_choose_from - 1)

        examples_compiled, real_example_idx = "", 0
        for i in range(1, num_examples_to_choose_from + 1):
            if i == ground_truth_placement:
                examples_compiled += (
                    f"Question {i}:\n"
                    + f"##Context: {data_point["context"]}\n"
                    + f"##Question: {data_point["question"]}\n"
                    + f"##Answer: {data_point["answer"]}\n"
                )
            else:
                examples_compiled += (
                    f"Question {i}:\n"
                    + f"##Context: {random_sample[real_example_idx]["context"]}\n"
                    + f"##Question: {random_sample[real_example_idx]["question"]}\n"
                    + f"##Answer: {random_sample[real_example_idx]["answer"]}\n"
                )
                real_example_idx += 1

        return hotpotqa_ir_prompt.format(
            k=num_examples_to_choose_from,
            examples=examples_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return hotpotqa_ir_system_prompt.format(k=num_examples_to_choose_from)

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
            f"##Question: {generated_datum["question"]}\n"
            + f"##Reason: {generated_datum["reason"]} "
            + f"##Answer: {generated_datum["answer"]}"
        )


class HotpotQAGenerativeEvalTask(GenerativeEvalTask):
    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        return [(0, HotpotQADataGenerationTask())]

    def prep_for_training(
        self, generated_content: Dict[int, List[Dict[str, str]]], output_filename: str
    ):
        generated_content = generated_content[0]
        finetuning_data = []

        for i in range(len(generated_content)):
            try:
                context = generated_content[i]["context"]
                question = "Question: {}\nContext: {}\n".format(
                    generated_content[i]["question"], context
                )
                question += "Think step by step then provide your single word or phrase answer at the end saying 'Answer:'."

                point = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant tasked with answering questions given a set of context. Think step by step then provide your single word or phrase answer at the end saying 'Answer:'.",
                        },
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": generated_content[i]["reason"]
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


hotpotqa_static_examples = [
    {
        "context": 'Coke Kahani (Urdu: کوک کہانی\u200e ) is a 2012 Pakistani comedy drama sitcom directed by Mehreen Jabbar broadcasting on Broadcast syndication.\n Sitcom is written by Syed Mohammad Ahmed and Yasir Rana, starring Sonia Rehman, Faisal Rehman, Syra Yousuf, Syed Mohammad Ahmed, Yasir Hussain, Ahmed Zeb, Shamim Hilali.\n Sitcom was first aired on 3 November 2012.\nAdnan Siddiqui (Urdu: \u200e ) is a Pakistani actor and model who has worked in Lollywood and Hollywood and also made his debut in Bollywood with the Hindi film "Mom" (2017).\n He has appeared in many commercials and drama serials, including "Uroosa", "Pal Do Pal", "Meri Adhoori Mohabbat", "Meri Zaat Zara-e-Benishan", "Doraha", "Hawa Rait Aur Aangan", "Choti Si Kahani", "Vasl" and "Parsa".\n Siddiqui first started his filming career in the 1990s; he became notable for being cast in the popular drama "Uroosa" and one of the famous travel reality shows of the time "Gulls & Guys" directed by Shoaib Mansoor.\n In 2002, he was nominated for Best Actor (TV) in the Lux Style Awards.\n He also played a small role alongside Angelina Jolie and Irrfan Khan in the 2007 film "A Mighty Heart".\n In 2010, Siddique won Best Supporting Actor Award for Ishq Junoon Deewangi on Pakistan Media Award.\n He made his first debut in Pakistani film Yalghaar\nFamily Front (Urdu: فیملی فرنٹ\u200e ) was a 1997 Pakistani comedy drama sitcom.\n It was broadcast by the Pakistan Television Corporation (PTV World, now PTV News).\n Sitcom was directed by Waseem Abbas and written by Muhammad Younis Butt, starring Saba Hameed, Samina Ahmad, Waseem Abbas, Mira Hashmi, "Iram Hassan", Shahzad Nasim and Naseem Vicky.\n This comedy show became highly popular among the people and ran for many TV seasons.\nGol Chakkar is a 2012 Pakistani comedy film directed by Shahbaz Shigri, produced and written by Aisha Linnea Akhtar.\n Film features Ali Rehman Khan, Hasan Bruun Akhtar, Usman Mukhtar, Salmaan Ahmed Shaukat, Uzair Jaswal, Adil Gul, Saboor Pasha, Asad Ali Shigri and special appearance by Shahana Khan Khalil.\n This film is a sequel to "Sole Search" on the life of a character Candy Bhai from the earlier version.\n Candy Bhai, along with some new characters, gets into trouble when the boys decide to head over to Rawalpindi.\nYasir Hussain is a Pakistani actor and writer from Islamabad best known for his comic roles.\nAs the City Sleeps is the debut album by gothic rock band Mary and the Black Lamb.\n The album is about different experiences the band members have gone through.\n Most of which are based on true stories.\n All the songs on the album were written by Lindz Riot with the exception of, Stranger where Matt Kelly helped write and arrange it, and Silence which Nixon helped write.\nLahori Gate (Urdu: لاہوری گیٹ\u200e ) was a Pakistani comedy drama serial, aired on PTV Home.\n Serial\'s stars were Iftikhar Thakur, Sardar Kamal, Naseem Vicky, Abid Kashmiri, Waseem Abbas and Qavi Khan.\n This show is written by Waseem Abbas who also acts in it.\n He is a son of Pakistan\'s late famous film playback singer Inayat Hussain Bhatti.\n Shahid Aziz is the director of this comedy drama.\nRaju Rocket is a 2012 Pakistani comedy drama serial aired on the Hum TV.\n Serial was first aired on 27 August 2012; and is directed by "Saima Waseem" and written by "Imran Ali Safir", starring Danish Taimoor, Sumbul Iqbal, Madiha Rizvi, Rubina Ashraf, Nadia Afghan and Qazi Wajid.\nPakistani English literature refers to English literature that has been developed and evolved in Pakistan, as well as by members of the Pakistani diaspora who write in the English language.\n English is one of the official languages of Pakistan (the other being Urdu) and has a history going back to the British colonial rule in South Asia (the British Raj); the national dialect spoken in the country is known as Pakistani English.\n Today, it occupies an important and integral part in modern Pakistani literature.\n Dr. Alamgir Hashmi introduced the term "Pakistani Literature [originally written] in English" with his "Preface" to his pioneering book "Pakistani Literature: The Contemporary English Writers" (New York, 1978; Islamabad, 1987) as well as through his other scholarly work and the seminars and courses taught by him in many universities since 1970\'s.\n It was established as an academic discipline in the world following his lead and further work by other scholars, and it is now a widely popular field of study.\nGoogly Mohalla was a 36-episodic Cricket World Cup 2015 special Pakistani comedy drama with a Romantic glimpse.\n It starred "Jahanzeb Khan" and "Aimen Khan" as main leads, while "Hina Dilpazir" played a supporting role.\n',
        "question": 'What Pakistani actor and writer from Islamabad helped write for the 2012 Pakistani comedy drama sitcom, "Coke Kahani"?',
        "reason": 'Yasir Hussain is mentioned in the passage ##begin_quote## Yasir Hussain is a Pakistani actor and writer from Islamabad best known for his comic roles. ##end_quote## as a Pakistani actor and writer from Islamabad. Although the passage ##begin_quote## Sitcom is written by Syed Mohammad Ahmed and Yasir Rana, starring Sonia Rehman, Faisal Rehman, Syra Yousuf, Syed Mohammad Ahmed, Yasir Hussain, Ahmed Zeb, Shamim Hilali.  ## end_quote does not explicitly state Yasir Hussain\'s involvement in writing "Coke Kahani," it mentions Yasir Hussain’s involvement it is the closest and most logical association within the context. Hence, Yasir Hussain is inferred as the relevant individual.',
        "answer": "Yasir Hussain",
    },
    {
        "context": 'The ZERO candy bar, introduced in 1920, is a candy bar composed of a combination of caramel, peanut and almond nougat covered with a layer of white fudge (a.k.a. white chocolate fudge).\n Its outwardly white color, an unusual color for a candy bar, has become its trademark.\nPseudomonas mandelii is a fluorescent, Gram-negative, rod-shaped bacterium isolated from natural spring waters in France.\n Based on 16S rRNA analysis, "P. mandelii" has been placed in the "P. fluorescens" group.\nCanfield\'s Diet Chocolate Fudge soda is a zero-calorie, aspartame-sweetened carbonated soft drink canned and distributed by the A.J. Canfield Company of Elgin, Illinois, USA, a division of Select Beverages.\n Production for the midwestern United States is handled by the American Bottling Company, a subsidiary of Dr Pepper Snapple Group and distribution by Kehe Foods of Chicago.\nBorup Fiord Pass is a glacier-carved valley on Ellesmere Island in Nunavut, Canada.\n The valley contains a natural spring which carries fluids from the subsurface to the surface, sometimes passing through the glacial ice in the process.\n The spring is the only known place where sulfur from a natural spring is deposited over ice.\n At the Borup Fiord Pass spring, hydrogen sulphide gas in the water is converted to stable deposits of either elemental sulfur, the most common material in the deposit, or gypsum.\n The process by which hydrogen sulfide becomes sulfur is complex, and most often occurs when microbes, like bacteria, are present.\nA fudge cookie is a cookie that is prepared with fudge or that has the flavor, consistency or texture of fudge.\n Chocolate fudge cookies are a variety, along with other fudge flavors, such as peanut butter fudge.\nThe A.J. Canfield Company produces and bottles soda beverages including Canfield\'s Diet Chocolate Fudge, primarily in the Chicago area and was founded in 1924.\n Production for the midwestern United States is handled by the American Bottling Company, a subsidiary of Dr Pepper Snapple Group and distribution by Kehe Foods of Chicago.\nSpaw Sunday, or Spa Sunday, is a celebration held on the first Sunday in May and peculiar to Yorkshire and Lancashire.\n It is focused on local holy wells or spas whose spring waters are believed to have restorative or healing properties only on that day.\n Celebrations usually include a short pilgrimage from the local church to the spring, and a blessing of the waters by the clergy, after which the crowd take turns to smell or taste the usually highly sulphurous waters.\n Traditionally liquorice was steeped in a cup of collected water, or shaken in a bottle, to sweeten the taste.\n Though it is not officially recommended to drink the waters, watching others react to the strong taste is part of the spectacle.\n Dock pudding is served at the Calderdale events.\nMagic Shell is a dessert product produced by Smucker’s.\n It is a syrup that quickly hardens into a crispy shell when poured onto a cold surface, which is the origin of the product\'s name.\n The syrup is primarily designed for use on ice cream.\n It comes in several flavours, including chocolate, caramel, chocolate fudge, cupcake, cherry, and smores in addition to two unique flavours—One, with chocolate, caramel, and pecans which the company calls "Turtle Delight", and a flavour based upon the candy bar Twix, Hersheys, and Reeses.\nDiet Dr Pepper is a no-calorie Dr Pepper that was first introduced in 1986.\n This version replaces the high fructose corn syrup found in the original with aspartame.\n Diet Cherry Chocolate Dr Pepper (2007–2008) was introduced as a limited edition flavor on November 21, 2007.\n It was discontinued in April 2008.\n It became available in Canada in early January 2008.\n A nondiet version was never created.\n The taste is similar to Canfield\'s Diet Cherry Chocolate Fudge Soda, but with the distinctive Dr Pepper flavor.\n It was featured in the song "Cherry Chocolate Rain" by YouTube celebrity Tay Zonday.\n Upon ceasing production, it was replaced by Cherry Vanilla Dr Pepper.\nVolvic is a brand of mineral water.\n Its source is Clairvic Spring, Auvergne Regional Park just to the north of the Puy de Dôme in France.\n',
        "question": "Are both Volvic and Canfield's Diet Chocolate Fudge natural spring waters ?",
        "reason": "The context provides information on both Volvic and Canfield's Diet Chocolate Fudge. The passage ##begin_quote## Volvic is a brand of mineral water. Its source is Clairvic Spring, Auvergne Regional Park just to the north of the Puy de Dôme in France. ##end_quote## specifies that Volvic is a mineral water and its source is a natural spring. In contrast, the passage about Canfield's Diet Chocolate Fudge states ##begin_quote## Canfield's Diet Chocolate Fudge soda is a zero-calorie, aspartame-sweetened carbonated soft drink canned and distributed by the A.J. Canfield Company ##end_quote## indicating that it is a soda and not water from a natural spring. Therefore, Volvic is a natural spring water, but Canfield's Diet Chocolate Fudge is not.",
        "answer": "no",
    },
    {
        "context": "1st Word and 1st Word Plus are word processors developed by GST Computer Systems in the 1980s.\n The original package, 1st Word, was given away free with all Atari STs.\n The later 1st Word Plus was sold by GST and was more advanced.\n Atari ST disk magazine ST News was written entirely and exclusively using 1st Word and, later, 1st Word Plus.\n The first Volume (1986) was distributed as a plain 1st Word .\nDOC file, after that a custom shell was produced that enabled the 1st Word documents to be displayed in a userfriendly disk magazine shell.\nArm Holdings (Arm) is a British multinational semiconductor and software design company, owned by SoftBank Group and its Vision Fund.\n Headquartered in Cambridge, United Kingdom, its primary business is in the design of Arm processors (CPUs), although it also designs software development tools under the DS-5, RealView and Keil brands, as well as systems and platforms, system-on-a-chip (SoC) infrastructure and software.\n It is considered to be market dominant for processors in mobile phones (smartphones or otherwise) and tablet computers.\n The company is one of the best-known 'Silicon Fen' companies.\nThe e200 core is developed from the MPC5xx family processors, which in turn is derived from the MPC8xx core in the PowerQUICC SoC processors.\n e200 adheres to the Power ISA v.2.03 as well as the previous \"Book E\" specification.\n All e200 core based microprocessors are named in the MPC55xx and MPC56xx/JPC56x scheme, not to be confused with the MPC52xx processors which is based on the PowerPC e300 core.\nOMAP (Open Multimedia Applications Platform) is a series of image/video processors developed by Texas Instruments.\n They are a category of proprietary system on chips (SoCs) for portable and mobile multimedia applications.\n OMAP devices generally include a general-purpose ARM architecture processor core plus one or more specialized co-processors.\n Earlier OMAP variants commonly featured a variant of the Texas Instruments TMS320 series digital signal processor.\nAnalog Devices, Inc., also known as ADI or Analog, is an American multinational semiconductor company specializing in data conversion and signal processing technology, headquartered in Norwood, Massachusetts.\n In 2012, Analog Devices led the worldwide data converter market with a 48.5% share, according to analyst firm Databeans.\nIntel Corporation (also known as Intel, stylized as intel) is an American multinational corporation and technology company headquartered in Santa Clara, California (colloquially referred to as \"Silicon Valley\") that was founded by Gordon Moore (of Moore's law fame) and Robert Noyce.\n It is the world's second largest and second highest valued semiconductor chip makers based on revenue after being overtaken by Samsung, and is the inventor of the x86 series of microprocessors, the processors found in most personal computers (PCs).\n Intel supplies processors for computer system manufacturers such as Apple, Lenovo, HP, and Dell.\n Intel also manufactures motherboard chipsets, network interface controllers and integrated circuits, flash memory, graphics chips, embedded processors and other devices related to communications and computing.\nZet is a clone x86 processor where its machine code compatible with x86 processors developed as an effort to make open-hardware processor.\nXetal is the name of a family of non commercial massively parallel processors developed within Philips Research.\n.\nThe XAP processor is a RISC processor architecture developed by Cambridge Consultants since 1994.\n XAP processors are a family of 16-bit and 32-bit cores, all of which are intended for use in an application-specific integrated circuit or ASIC chip design.\n XAP processors were designed for use in mixed-signal integrated circuits for sensor or wireless applications including Bluetooth, ZigBee, GPS, RFID or Near Field Communication chips.\n Typically these integrated circuits are used in low cost, high volume products that are battery-powered and must have low energy consumption.\n There are other applications where XAP processors have been used to good effect, such as wireless sensor networks and medical devices, e.g. hearing aids.\nThe Blackfin is a family of 16- or 32-bit microprocessors developed, manufactured and marketed by Analog Devices.\n The processors have built-in, fixed-point digital signal processor (DSP) functionality supplied by 16-bit Multiply–accumulates (MACs), accompanied on-chip by a small microcontroller.\n It was designed for a unified low-power processor architecture that can run operating systems while simultaneously handling complex numeric tasks such as real-time H.264 video encoding.\n There are several hardware development kits for the Blackfin.\n Open-source operating systems for the Blackfin include uClinux.\n",
        "question": "Blackfin is a family of processors developed by the company that is headquartered in what city?",
        "reason": "The document ##begin_quote## The Blackfin is a family of 16- or 32-bit microprocessors developed, manufactured and marketed by Analog Devices. ##end_quote## establishes that the Blackfin processors are developed by Analog Devices. The document ##begin_quote## Analog Devices, Inc., also known as ADI or Analog, is an American multinational semiconductor company specializing in data conversion and signal processing technology, headquartered in Norwood, Massachusetts. ##end_quote## provides the location of the headquarters of Analog Devices. Therefore, the company that developed the Blackfin processors is headquartered in Norwood, Massachusetts.",
        "answer": "Norwood, Massachusetts",
    },
]

hotpotqa_personas = [
    {
        "persona": "a historian who writes questions that focus on historical context, cause and effect, and key figures in the text."
    },
    {
        "persona": "a detective who crafts questions that require careful attention to detail, inference, and logical reasoning based on textual evidence."
    },
    {
        "persona": "a philosopher who asks deep, open-ended questions that probe themes, ethics, and abstract ideas within the text."
    },
    {
        "persona": "a teacher who creates straightforward comprehension questions that check for basic understanding and recall."
    },
    {
        "persona": "a journalist who writes questions that emphasize the ‘who, what, when, where, why, and how’ of the text, seeking clear, factual answers."
    },
    {
        "persona": "a critic who develops questions that encourage analysis of the text’s structure, style, and effectiveness in conveying its message."
    },
    {
        "persona": "a scientist who frames questions that highlight logical consistency, cause-and-effect relationships, and data interpretation in the text."
    },
    {
        "persona": "a psychologist who generates questions that explore character motivations, emotions, and relationships within the text."
    },
    {
        "persona": "a lawyer who constructs questions that demand precise evidence from the text to support claims, much like building an argument in court."
    },
    {
        "persona": "a storyteller who focuses on narrative elements, asking questions about plot development, symbolism, and storytelling techniques."
    },
]


hotpotqa_persona_prompt = """You are {persona}
Provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be a single word or phrase, not a sentence. The context is provided to you in the following format:
"##Context:
[sentence_1]
[sentence_2]
..."

You should provide the question and reasoning in the following format:
"##Question: [question]
##Reason: [reasoning] ##Answer: [final_answer]"
Use ##begin_quote## and ##end_quote## to denote quotations from the context in your reasoning.

Here are some examples of contexts, questions, and reasoning:
{examples}

Now it's your turn. Here is the context:
{context}
Now, generate a question and corresponding reasoning and answer, following the format in the examples. Start your response with the question."""

hotpotqa_base_prompt = """Here are a few examples of contexts and corresponding questions that can be answered using information in just a few sentences in that context, as well as the reasoning and answer to those questions. The answer should be a single word or phrase, not a sentence. Note how ##begin_quote## and ##end_quote## are used to denote quotations from the context in the reasoning.
{examples}

EXAMPLE START
##Context:
{context}
"""

hotpotqa_instruct_fewshot_prompt = """Provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be a single word or phrase, not a sentence. The context is provided to you in the following format:
"##Context:
[sentence_1]
[sentence_2]
..."

You should provide the question and reasoning in the following format:
"##Question: [question]
##Reason: [reasoning] ##Answer: [final_answer]"
Use ##begin_quote## and ##end_quote## to denote quotations from the context in your reasoning.

Here are some examples of contexts, questions, and reasoning:
{examples}

Now it's your turn. Here is the context:
{context}
Now, generate a question and corresponding reasoning and answer, following the format in the examples. Start your response with the question."""

hotpotqa_refine_prompt = """Improve the given question and corresponding reasoning. The question should be a realistic question someone may ask for the given context, and the reasoning and answer should be correct. You should edit the question and reasoning minimally to be more coherent. The answer should be a single word or phrase, not a sentence. Do not change the theme of the question or reasoning. The context is provided to you in the following format:
"##Context:
[sentence_1]
[sentence_2]
..."

You should provide the edited question and reasoning in the following format:
"##Question: [question]
##Reason: [reasoning] ##Answer: [final_answer]"
##begin_quote## and ##end_quote## should be used to denote quotations from the context. Here are some examples of contexts, questions, and reasoning:
{examples}

Now it's your turn. Here is the context:
{context}
And here is the question and reasoning for you to edit:
##Question: {question}
##Reason: {reason} ##Answer {answer}
Provide only the improved question, reasoning, and answer in the given format. Do not include any commentary or notes. Start your response with the question."""

hotpotqa_instruct_sequential_prompt = """Your job is to provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. You must generate a new question that could be asked from the given context that is substantially different from the previously generated examples. The answer should be a single word or phrase, not a sentence.

The context is provided to you in the following format:
"##Context:
[sentence_1]
[sentence_2]
..."

You should provide the question and reasoning in the following format:
"##Question: [question]
##Reason: [reasoning] ##Answer: [final_answer]"
Use ##begin_quote## and ##end_quote## to denote quotations from the context in your reasoning.

Here are the previously generated examples:
{examples}

Now it's your turn. Here is the context:
{context}

Now, generate a question and corresponding reasoning and answer, following the format in the examples. Start your response with the question."""

hotpotqa_instruct_in_one_prompt = """Your job is to provide {num} examples of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be a single word or phrase, not a sentence.

The context is provided to you in the following format:
"##Context:
[sentence_1]
[sentence_2]
..."

You should provide each question and reasoning in the following format:
"##Question: [question]
##Reason: [reasoning] ##Answer: [final_answer]"
Use ##begin_quote## and ##end_quote## to denote quotations from the context in your reasoning.

Here are some examples:
{examples}

Now it's your turn. Here is the context:
{context} 

Generate {num} different questions and reasonings and answers following this format. Your question should be different in content from the examples. Make sure to only provide only the question, reasoning, and answer, starting each example with the question. Delimit the end of an example with the phrase "END OF EXAMPLE" (all caps) on a new line."""

hotpotqa_ir_system_prompt = """You are an expert at evaluating question and answer pairs that can be answered given some context.

You will be shown {k} examples. Each examples consists of some context, a question, and an answer. All but one of them is generated from a high quality AI while one of them is of low quality.

Your task is to identify which message (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure and clarity
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each example carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

hotpotqa_ir_prompt = """Here are {k} examples. One of them is of low quality. Please identify which one:

{examples}

Analyze each example and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
