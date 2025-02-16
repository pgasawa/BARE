# BARE Development

## Setup: 
Requires Python >= 3.12.

- Clone the repo and ```cd``` into the directory
- Create a new virtual environment: ```python3 -m venv venv```
- Activate the virtual environment: ```source venv/bin/activate```
- Run ```make install```
- Create a ```.env``` configuration file in the root dir with model provider API keys following ```litellm``` specs
    - Example: ```OPENAI_API_KEY=your_api_key```
    - Alternatively, you can set the secrets in your shell.
- Run ```python -m experiments.build_pubmedqa_dataset``` to setup the PubMedQA dataset.

## Development: 
- At a high level, the initial data is generated per user specifications in ```experiments/synthetic_data_runner.py```. The resulting data is then converted into a format that is compatible with standard SFT for most fine-tuning providers in ```experiments/evals/prep_sft_data.py```. After training a model on the SFT data, you can evaluate it using the eval files in ```experiments/evals```. For classification tasks, a BERT model can be directly trained in ```experiments/evals/classification_eval_runner.py```. Analyses on diversity and quality can be performed in ```experiments/analysis/analysis_runner.py```.
- To create synthetic data, from the root, run ```python -m experiments.synthetic_data_runner```. You should specify the configuration for the data/task in the ```experiments/synthetic_data_runner.py``` file.
- To create a new generation task and add your own prompts, follow the examples in ```src/tasks```.
- Run ```make format``` before commits to automatically format and lint.
- Run ```make update-requirements``` to update requirements.
