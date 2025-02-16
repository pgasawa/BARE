import os
import pickle

from pathlib import Path

from src.tasks import GenerativeEvalTask


def convert_to_training_jsonl(
    source_pkl_file: str,
    generative_eval_task: GenerativeEvalTask,
    output_jsonl_file: str,
):
    with open(source_pkl_file, "rb") as f:
        generated_data = pickle.load(f)

    output_path = Path(output_jsonl_file)
    os.makedirs(output_path.parent.absolute(), exist_ok=True)
    generative_eval_task.prep_for_training(generated_data, output_jsonl_file)


if __name__ == "__main__":
    # Importing example task
    from src.tasks import GSM8KGenerativeEvalTask as ExampleTask

    # sft data prep example
    experiments = [
        {
            "source_pkl_file": "generated_data/gsm8k/bare.pkl",
            "generative_eval_task": ExampleTask(),
            "output_jsonl_file": "finetuning_data/gsm8k/bare.jsonl",
        },
    ]
    for parameter_set in experiments:
        convert_to_training_jsonl(**parameter_set)
