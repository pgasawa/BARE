import random
import json
import pandas as pd
from datasets import load_from_disk
from typing import Dict, List, Tuple

from src.tasks import DataGenerationTask, GenerativeEvalTask, DATASET_CACHE_DIR


class PubMedQADataGenerationTask(DataGenerationTask):
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
                pubmedqa_static_examples[:k], instruct=False
            )
            return pubmedqa_base_prompt.format(
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
                pubmedqa_static_examples[:k], instruct=True
            )
            return pubmedqa_instruct_fewshot_prompt.format(
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
            pubmedqa_static_examples + prev_examples, instruct=True
        )
        return pubmedqa_instruct_sequential_prompt.format(
            examples=examples_str, context=grounding["context"]
        )

    def instruct_in_one_prompt(self, grounding: Dict[str, str], num: int) -> str:
        examples_str = self.format_prompt_examples(
            pubmedqa_static_examples, instruct=True
        )
        return pubmedqa_instruct_in_one_prompt.format(
            num=num, examples=examples_str, context=grounding["context"]
        )

    def split_instruct_in_one(self, raw_response: str) -> List[str]:
        return raw_response.split("END OF EXAMPLE")[:-1]

    def instruct_persona_prompt(self, grounding: Dict[str, str]) -> str:
        examples_str = self.format_prompt_examples(
            pubmedqa_static_examples, instruct=True
        )
        return pubmedqa_persona_prompt.format(
            examples=examples_str,
            persona=grounding["persona.persona"],
            context=grounding["context"],
        )

    def instruct_refinement_prompt(
        self, base_content: Dict[str, str], k: int = 3
    ) -> str:
        examples_str = self.format_prompt_examples(
            pubmedqa_static_examples[:k], instruct=True
        )
        return pubmedqa_refine_prompt.format(
            examples=examples_str,
            context=base_content["context"],
            question=base_content["question"],
            reason=base_content["reason"],
            answer=base_content["answer"],
        )

    # general prompt methods
    def build_fewshot_examples(self):
        if self.candidate_examples is None:
            dataset = load_from_disk(DATASET_CACHE_DIR + "/pub_med_qa_distractors")

            self.candidate_examples = [
                {"context": c, "question": q, "answer": a}
                for c, q, a in zip(
                    dataset["context"],
                    dataset["question"],
                    dataset["answer"],
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
        dataset = load_from_disk(DATASET_CACHE_DIR + "/pub_med_qa_distractors")
        df = pd.DataFrame()
        df["context"] = dataset["context"]
        df = df.sample(n=num, random_state=seed)
        grounding = df["context"].tolist()
        formatted_grounding = [{"context": g} for g in grounding]
        return formatted_grounding

    def load_personas(self) -> List[Dict[str, str]]:
        return pubmedqa_personas

    def separate_grounding(self, content):
        return {"context": content["context"]}

    def extract_content(
        self, raw_response: str, grounding: Dict = None
    ) -> Dict[str, str]:
        question = raw_response.split("##Reason:")[0]
        reason_answer = raw_response.split("##Reason:")[1]
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

        return pubmedqa_ir_prompt.format(
            k=num_examples_to_choose_from,
            examples=examples_compiled,
        )

    def get_quality_system_prompt(self, num_examples_to_choose_from: int = 4) -> str:
        return pubmedqa_ir_system_prompt.format(k=num_examples_to_choose_from)

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


class PubMedQAGenerativeEvalTask(GenerativeEvalTask):
    def constituent_tasks(self) -> List[Tuple[int, DataGenerationTask]]:
        return [(0, PubMedQADataGenerationTask())]

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


pubmedqa_static_examples = [
    {
        "context": """INTRODUCTION: Halofantrine is a newly developed antimalarial drug used for the treatment of Plasmodium falciparum malaria. The introduction of this drug has been delayed because of its possible side effects, and due to insufficient studies on adverse reactions in humans. There have been no studies investigating its effect on hearing.\nMETHODS: Thirty guinea pigs were divided into three groups: a control group, a halofantrine therapeutic dose group and a halofantrine double therapeutic dose group. One cochlea specimen from each animal was stained with haematoxylin and eosin and the other with toluidine blue.\nRESULTS: No changes were detected in the control group. The halofantrine therapeutic dose group showed loss and distortion of inner hair cells and inner phalangeal cells, and loss of spiral ganglia cells. In the halofantrine double therapeutic dose group, the inner and outer hair cells were distorted and there was loss of spiral ganglia cells.\n\nPURPOSE: This study was conducted to investigate the expression and functional impact of the proto-oncogene c-kit in uveal melanoma.\nMETHODS: Based on immunohistochemical (IHC) study of paraffin-embedded specimens from 134 uveal melanomas and Western blot analysis on eight fresh-frozen samples the expression of c-kit in uveal melanoma was studied. Furthermore, the phosphorylation of c-kit and the impact of the tyrosine kinase inhibitor STI571 was examined in the three uveal melanoma cell lines OCM-1, OCM-3, and 92-1.\nRESULTS: Eighty-four of 134 paraffin-embedded samples and six of eight fresh-frozen samples expressed c-kit. c-Kit was strongly expressed and tyrosine phosphorylated in cultured uveal melanoma cells compared with cutaneous melanoma cells. Moreover, in contrast to cutaneous melanoma cell lines c-kit maintained a high phosphorylation level in serum-depleted uveal melanoma cells. No activation-related mutations in exon 11 of the KIT gene were found. On the contrary, expression of the stem cell growth factor (c-kit ligand) was detected in all three uveal melanoma cell lines, suggesting the presence of autocrine (paracrine) stimulation pathways. Treatment of uveal melanoma cell lines with STI571, which blocks c-kit autophosphorylation, resulted in cell death. The IC(50) of the inhibitory effects on c-kit phosphorylation and cell proliferation was of equal size and less than 2.5 microM.\n\nBACKGROUND AND AIMS: The hypothesis was tested that pectin content and methylation degree participate in regulation of cell wall mechanical properties and in this way may affect tissue growth and freezing resistance over the course of plant cold acclimation and de-acclimation.\nMETHODS: Experiments were carried on the leaves of two double-haploid lines of winter oil-seed rape (Brassica napus subsp. oleifera), differing in winter survival and resistance to blackleg fungus (Leptosphaeria maculans).\nKEY RESULTS: Plant acclimation in the cold (2 degrees C) brought about retardation of leaf expansion, concomitant with development of freezing resistance. These effects were associated with the increases in leaf tensile stiffness, cell wall and pectin contents, pectin methylesterase (EC 3.1.1.11) activity and the low-methylated pectin content, independently of the genotype studied. However, the cold-induced modifications in the cell wall properties were more pronounced in the leaves of the more pathogen-resistant genotype. De-acclimation promoted leaf expansion and reversed most of the cold-induced effects, with the exception of pectin methylesterase activity.\n\nBACKGROUND: Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.\nRESULTS: The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\n\nBACKGROUND: The aim of this study was to analyze the properties of the immune cell microenvironment of regional lymph nodes (LNs) positive for lung cancer.\nMETHODS: Twenty-four patients operated on for stages T1 and T2 of the NSCLC, were enrolled in the study. Peripheral blood and LN tissue were obtained from different lymph node sites and levels. As a control, LN tissue was taken from patients diagnosed with emphysema or pneumothorax. The cells from randomly chosen LN were tested by multi-color flow cytometry. Separate portions of LN were snap-frozen and examined for the presence of cytokeratin positive cells (CK). Propensity for apoptosis, level of TCR zeta chain expression of T cells and the number and maturation status of dendritic cells were confronted with the presence of CK-positive cells.\nRESULTS: The presence of metastases correlated with the downregulation of TCR zeta, especially CD8(+) T cells. The most striking feature was the reduction in the number of myeloid CD11c(+) dendritic cells in the LN of patients with LN metastases. This could be a reflection of the immunodeficient state observed in lung cancer patients. Even in the absence of metastases in the regional LN, the same type of changes in the LN microenvironment were observed in those LN located nearer the primary tumor.\n""",
        "question": """Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?""",
        "reason": """The results in the document depict mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. Specifically, the treatment they used ##begin_quote## This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells. ##end_quote##. To the best of our knowledge, this is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, the findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.""",
        "answer": "yes",
    },
    {
        "context": """METHODS: The records of 465 patients with an established diagnosis of age related macular degeneration who had attended a specialist macular clinic between 1990 and 1998 were scrutinised. A full clinical examination and standardised refraction had been carried out in 189 of these cases on a minimum of two occasions. Cases were looked for where an improvement of one or more lines of either distance or near acuity was recorded in the eye unaffected by macular disease. In each one of these cases the improvement in visual acuity could not be attributed to treatment of other existing pathology.\nRESULTS: 12 such cases were detected. In nine of these the eye showing improvement of acuity had a history of amblyopia. The mean improvement in distance and near acuity in amblyopic eyes by 12 months was 3.3 and 1.9 lines logMAR respectively. The improvement in acuity generally occurred between 1 and 12 months from baseline and remained stable over the period of follow up.\n\nPURPOSE: The precise correction of refractive error is especially important in young adults. It is unclear whether cycloplegic refraction is necessary in this age group. The purpose of this study was to compare the non-cycloplegic and cycloplegic spherical equivalent (SE) refractive error measured in young adults.\nMETHODS: This was a prospective study of 1400 eyes (n\u2009=\u2009700) of enlisted soldiers aged 18 to 21\xa0years who were consecutively evaluated in an outpatient army ophthalmology clinic. One drop of cyclopentolate 1\xa0% was installed twice 10\xa0min apart, and cycloplegic refraction was performed in both eyes 40\xa0min later using an auto-refractor. The difference between non-cycloplegic and cycloplegic refractive measurements was analyzed.\nRESULTS: The mean difference in SE between non-cycloplegic and cycloplegic measurements was 0.68\u2009±\u20090.83\xa0D (95\xa0% CI, 0.64-0.72). Significantly greater differences were observed in hypermetropes than myopes (1.30\u2009±\u20090.90\xa0D versus 0.46\u2009±\u20090.68\xa0D, p\u2009<\u20090.001). Moderate hypermetropes (2 to 5\xa0D) demonstrated significantly greater refractive error than mild (0.5 to 2\xa0D) or severe (>5\xa0D) hypermetropes (1.71\u2009±\u20091.18\xa0D versus 1.19\u2009±\u20090.74\xa0D and 1.16\u2009±\u20091.08\xa0D respectively, p\u2009<\u20090.001).\n\nPURPOSE: To investigate whether the Patient Health Questionnaire-9 (PHQ-9) possesses the essential psychometric characteristics to measure depressive symptoms in people with visual impairment.\nMETHODS: The PHQ-9 scale was completed by 103 participants with low vision. These data were then assessed for fit to the Rasch model.\nRESULTS: The participants' mean +/- standard deviation (SD) age was 74.7 +/- 12.2 years. Almost one half of them (n = 46; 44.7%) were considered to have severe vision impairment (presenting visual acuity<6/60 in the better eye). Disordered thresholds were evident initially. Collapsing the two middle categories produced ordered thresholds and fit to the Rasch model (chi = 10.1; degrees of freedom = 9; p = 0.34). The mean (SD) items and persons Fit Residual values were -0.31 (1.12) and -0.25 (0.78), respectively, where optimal fit of data to the Rasch model would have a mean = 0 and SD = 1. Unidimensionality was demonstrated confirming the construct validity of the PHQ-9 and there was no evidence of differential item functioning on a number of factors including visual disability. The person separation reliability value was 0.80 indicating that the PHQ-9 has satisfactory precision. There was a degree of mistargeting as expected in this largely non-clinically depressed sample.\n\nPURPOSE: Recent studies have found a choroidal thickening in amblyopic eyes and suggested that there might be a relationship between the choroid and amblyopia. The present study aimed to evaluate the effect of a six-month treatment of amblyopia on choroidal thickness in anisometropic hyperopic amblyopic children.\nMETHODS: Thirty-two anisometropic hyperopic children with unilateral amblyopia were included in this prospective study. Subfoveal choroidal thickness was measured as the distance between the retinal pigment epithelium and the chorioscleral edge, by using spectral domain enhanced depth imaging optical coherence tomography. The treatment of amblyopia was performed based on the full correction of the refractive error with eyeglasses, a refractive adaptation phase and occlusion by patching the fellow eye.\nRESULTS: The mean visual acuity of the amblyopic eyes significantly increased from 0.35\u2009±\u20090.3 to 0.16\u2009±\u20090.2 logMAR after the treatment (p\u2009<\u20090.001). The mean initial choroidal thickness was significantly higher in the amblyopic eyes than in the fellow eyes (p\u2009=\u20090.019). There were no significant differences between the pre- and post-treatment mean choroidal thickness in the amblyopic eyes (p\u2009=\u20090.428) and in the fellow eyes (p\u2009=\u20090.343). The mean choroidal thickness was still higher in the amblyopic eyes than in the fellow eyes after the treatment (p\u2009=\u20090.006).\n\nBACKGROUND: Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with the Snellen E was compared to the Landolt C acuity.\nPATIENTS AND METHODS: 100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In the group of patients, the eyes with the lower visual acuity, and the right eyes of the healthy subjects, were evaluated.\nRESULTS: Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of the other groups were similar with only small differences between LR and SE.\n""",
        "question": """Landolt C and snellen e acuity: differences in strabismus amblyopia?""",
        "reason": """The document says ##begin_quote## Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small ##end_quote##. Using the charts described, there was only a slight overestimation of visual acuity by the Snellen E compared to the Landolt C, even in strabismus amblyopia, specifically ##begin_quote## The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia ##end_quote##. Thus the differences are too minor to be considered significant.""",
        "answer": "no",
    },
    {
        "context": """BACKGROUND: "America\'s Best Hospitals," an influential list published annually by U.S. News and World Report, assesses the quality of hospitals. It is not known whether patients admitted to hospitals ranked at the top in cardiology have lower short-term mortality from acute myocardial infarction than those admitted to other hospitals or whether differences in mortality are explained by differential use of recommended therapies.\nMETHODS: Using data from the Cooperative Cardiovascular Project on 149,177 elderly Medicare beneficiaries with acute myocardial infarction in 1994 or 1995, we examined the care and outcomes of patients admitted to three types of hospitals: those ranked high in cardiology (top-ranked hospitals); hospitals not in the top rank that had on-site facilities for cardiac catheterization, coronary angioplasty, and bypass surgery (similarly equipped hospitals); and the remaining hospitals (non-similarly equipped hospitals). We compared 30-day mortality; the rates of use of aspirin, beta-blockers, and reperfusion; and the relation of differences in rates of therapy to short-term mortality.\nRESULTS: Admission to a top-ranked hospital was associated with lower adjusted 30-day mortality (odds ratio, 0.87; 95 percent confidence interval, 0.76 to 1.00; P=0.05 for top-ranked hospitals vs. the others). Among patients without contraindications to therapy, top-ranked hospitals had significantly higher rates of use of aspirin (96.2 percent, as compared with 88.6 percent for similarly equipped hospitals and 83.4 percent for non-similarly equipped hospitals; P<0.01) and beta-blockers (75.0 percent vs. 61.8 percent and 58.7 percent, P<0.01), but lower rates of reperfusion therapy (61.0 percent vs. 70.7 percent and 65.6 percent, P=0.03). The survival advantage associated with admission to top-ranked hospitals was less strong after we adjusted for factors including the use of aspirin and beta-blockers (odds ratio, 0.94; 95 percent confidence interval, 0.82 to 1.08; P=0.38).\n\nBACKGROUND: It is unclear whether traveling long distances to high-volume centers would compensate for travel burden among patients undergoing rectal cancer resection.\nOBJECTIVE: The purpose of this study was to determine whether operative volume outweighs the advantages of being treated locally by comparing the outcomes of patients with rectal cancer treated at local, low-volume centers versus far, high-volume centers.\nDESIGN: This was a population-based study.\nSETTINGS: The National Cancer Database was queried for patients with rectal cancer.\nPATIENTS: Patients with stage II or III rectal cancer who underwent surgical resection between 2006 and 2012 were included.\nMAIN OUTCOME MEASURES: The outcomes of interest were margins, lymph node yield, receipt of neoadjuvant chemoradiation, adjuvant chemotherapy, readmission within 30 days, 30-day and 90-day mortality, and 5-year overall survival.\nRESULTS: A total of 18,605 patients met inclusion criteria; 2067 patients were in the long-distance/high-volume group and 1362 in the short-distance/low-volume group. The median travel distance was 62.6 miles for the long-distance/high-volume group and 2.3 miles for the short-distance/low-volume group. Patients who were younger, white, privately insured, and stage III were more likely to have traveled to a high-volume center. When controlled for patient factors, stage, and hospital factors, patients in the short-distance/low-volume group had lower odds of a lymph node yield ≥12 (OR = 0.51) and neoadjuvant chemoradiation (OR = 0.67) and higher 30-day (OR = 3.38) and 90-day mortality (OR = 2.07) compared with those in the long-distance/high-volume group. The short-distance/low-volume group had a 34% high risk of overall mortality at 5 years compared with the long-distance/high-volume group.\nLIMITATIONS: We lacked data regarding patient and physician decision making and surgeon-specific factors.\n\nOBJECTIVE: We compare 30-day and 180-day postadmission hospital mortality rates for all Medicare patients and those in three categories of cardiac care: coronary artery bypass graft surgery, acute myocardial infarction, and congestive heart failure. DATA SOURCES/\nCOLLECTION: Health Care Financing Administration (HCFA) hospital mortality data for FY 1989.\nSTUDY DESIGN: Using hospital level public use files of actual and predicted mortality at 30 and 180 days, we constructed residual mortality measures for each hospital. We ranked hospitals and used receiver operating characteristic (ROC) curves to compare 0-30, 31-180, and 0-180-day postadmission mortality.\nPRINCIPAL FINDINGS: For the admissions we studied, we found a broad range of hospital performance when we ranked hospitals using the 30-day data; some hospitals had much lower than predicted 30-day mortality rates, while others had much higher than predicted mortality rates. Data from the time period 31-180 days postadmission yield results that corroborate the 0-30 day postadmission data. Moreover, we found evidence that hospital performance on one condition is related to performance on the other conditions, but that the correlation is much weaker in the 31-180-day interval than in the 0-30-day period. Using ROC curves, we found that the 30-day data discriminated the top and bottom fifths of the 180-day data extremely well, especially for AMI outcomes.\n\nOBJECTIVES: To study the relationship between coronary angiography and in-hospital mortality in patients undergoing emergency surgery of the aorta without a history of coronary revascularization or coronary angiography before the onset of symptoms.\nBACKGROUND: In the setting of acute ascending aortic dissection warranting emergency aortic repair, coronary angiography has been considered to be desirable, if not essential. The benefits of defining coronary anatomy have to be weighed against the risks of additional delay before surgical intervention.\nMETHODS: Retrospective analysis of patient charts and the Cardiovascular Information Registry (CVIR) at the Cleveland Clinic Foundation.\nRESULTS: We studied 122 patients who underwent emergency surgery of the aorta between January 1982 and December 1997. Overall, in-hospital mortality was 18.0%, and there was no significant difference between those who had coronary angiography on the day of surgery compared with those who had not (No: 16%, n = 81 vs. Yes: 22%, n = 41, p = 0.46). Multivariate analysis revealed that a history of myocardial infarction (MI) was the only predictor of in-hospital mortality (relative risk: 4.98 95% confidence interval: 1.48-16.75, p = 0.009); however, coronary angiography had no impact on in-hospital mortality in patients with a history of MI. Furthermore, coronary angiography did not significantly affect the incidence of coronary artery bypass grafting (CABG) during aortic surgery (17% vs. 25%, Yes vs. No). Operative reports revealed that 74% of all CABG procedures were performed because of coronary dissection, and not coronary artery disease.\n\nAIMS: Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting the 30-day mortality to be 14-15%. The aim of this study was to analyse the 30-day mortality, age-related 30-day mortality and 1-year mortality following emergency laparotomy. We hope this will encourage prospective data collection, improvement of care and initiate strategies to establish best practice in this area.\nMETHODS: This was a retrospective study of patients who underwent emergency laparotomy from June 2010 to May 2012. The primary end point of the study was 30-day mortality, age-related 30-day mortality and 1-year all-cause mortality.\nRESULTS: 477 laparotomies were performed in 446 patients. 57% were aged<70 and 43% aged>70 years. 30-day mortality was 12, 4% in those aged<70 years and 22% in those>70 years (p<0.001). 1-year mortality was 25, 15% in those aged under 70 years and 38% in those aged>70 years (p<0.001).\n""",
        "question": """30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement?""",
        "reason": """Emergency laparotomy carries a high rate of mortality, ##begin_quote## 30-day mortality was 12, 4% in those aged<70 years and 22% in those>70 years (p<0.001). 1-year mortality was 25, 15% in those aged under 70 years and 38% in those aged>70 years (p<0.001). ##end_quote##, and more needs to be done to improve outcomes, particularly in this group. This could involve increasing acute surgical care manpower, early recognition of patients requiring emergency surgery, development of clear management protocols for such patients or perhaps even considering centralisation of emergency surgical services to specialist centres with multidisciplinary teams involving emergency surgeons and care of the elderly physicians in hospital and related community outreach services for post-discharge care.""",
        "answer": "maybe",
    },
]

pubmedqa_personas = [
    {
        "persona": "an epidemiologist investigating the association between lifestyle factors and disease outcomes."
    },
    {
        "persona": "a pharmaceutical researcher evaluating the efficacy of a new drug compared to existing treatments."
    },
    {
        "persona": "a public health official assessing the impact of vaccination programs on disease prevalence."
    },
    {
        "persona": "a clinical physician looking for evidence-based guidelines to improve patient care."
    },
    {
        "persona": "a biomedical scientist exploring the molecular mechanisms underlying a specific disease."
    },
    {
        "persona": "a health policy analyst studying the effects of healthcare interventions on patient outcomes."
    },
    {
        "persona": "a medical ethicist examining the risks and benefits of emerging medical technologies."
    },
    {
        "persona": "a nutritionist investigating the role of dietary components in chronic disease prevention."
    },
    {
        "persona": "a mental health researcher analyzing the effectiveness of therapeutic approaches for psychiatric disorders."
    },
    {
        "persona": "a sports medicine specialist evaluating the impact of physical activity on injury prevention and recovery."
    },
]

pubmedqa_persona_prompt = """You are {persona}
Provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be only yes, no, or maybe. The context is provided to you in the following format:
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

pubmedqa_base_prompt = """Here are a few examples of contexts and corresponding questions that can be answered using information in just a few sentences in that context, as well as the reasoning and answer to those questions. The answer should be only yes, no, or maybe. Note how ##begin_quote## and ##end_quote## are used to denote quotations from the context in the reasoning.
{examples}

EXAMPLE START
##Context:
{context}

##Question: """

pubmedqa_instruct_fewshot_prompt = """Provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be only yes, no, or maybe. The context is provided to you in the following format:
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

pubmedqa_refine_prompt = """Improve the given question and corresponding reasoning. The question should be a realistic question someone may ask for the given context, and the reasoning and answer should be correct. You should edit the question and reasoning minimally to be more coherent. The answer should be only yes, no, or maybe. Do not change the theme of the question or reasoning. The context is provided to you in the following format:
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

pubmedqa_instruct_sequential_prompt = """Your job is to provide an example of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. You must generate a new question that could be asked from the given context that is substantially different from the previously generated examples. The answer should be yes, no, or maybe.

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

pubmedqa_instruct_in_one_prompt = """Your job is to provide {num} examples of a question one could ask for a given context, and which would be answerable using information in just a few sentences in that context, as well as the corresponding reasoning and answer. The answer should be yes, no, or maybe.

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

Generate {num} different questions and reasonings and answers following this format. Your question should be different in content from the examples. Make sure to only provide only the question, reasoning, and answer. Start each example with the question. Delimit the end of an example with the phrase "END OF EXAMPLE" (all caps) on a new line."""

pubmedqa_ir_system_prompt = """You are an expert at evaluating question and answer pairs that can be answered given some context.

You will be shown {k} examples. Each examples consists of some context, a question, and an answer. All but one of them is generated from a high quality AI while one of them is of low quality.

Your task is to identify which message (1, 2, ..., {k}) appears to be of low quality. Consider factors like:

1. Differing natural language patterns and flow
2. Differing question structure and clarity
3. Context and specificity
4. Any subtle artifacts or unnatural patterns

Analyze each example carefully and explain your reasoning. End with 'Answer: [Question Number]' where Question Number is 1, 2, ..., {k}."""

pubmedqa_ir_prompt = """Here are {k} examples. One of them is of low quality. Please identify which one:

{examples}

Analyze each example and explain which one you think is of low quality. End with 'Answer: [Question Number]'."""
