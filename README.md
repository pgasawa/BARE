# BARE: Leveraging Base Language Models for Few-Shot Synthetic Data Generation

Authors: *Alan Zhu\*, Parth Asawa\*, Jared Quincy Davis, Lingjiao Chen, Boris Hanin, Ion Stoica, Joseph E. Gonzalez, Matei Zaharia*

*Equal contribution.

📜 Paper: [https://arxiv.org/abs/2502.01697](https://arxiv.org/abs/2502.01697)  
💻 Code: In this repo! (```bare/```)  
𝕏 Thread: [Post](https://x.com/pgasawa/status/1887201938607120592)

## 📋 TL;DR

- Instruct models → Higher quality, lower diversity.
- Base models → Higher diversity, lower quality.
- **BARE** → Best of both worlds.

## Problem

Instruct-tuned models are getting better at following instructions and ‘reasoning’ every day, but they’re shockingly poor at generating diverse responses. LLMs need diverse, high-quality synthetic data to train well, and we hypothesize this shortcoming in common methods hinders downstream performance. 

As a motivating example, when we sampled GPT-4o mini **100** times at temperature=1, we only got 4 distinct jokes (their quality can be debated):

```
1. Why did the scarecrow win an award? Because he was outstanding in his field!
2. Why don't scientists trust atoms? Because they make up everything!
3. Why don’t skeletons fight each other? They don’t have the guts!
4. Why do seagulls fly over the ocean? Because if they flew over the bay, they’d be bagels!
```

## 🐻 BARE

![image](assets/BARE.png)

We introduce Base-Refine (**BARE** 🐻), a method for combining base language models and instruction-tuned language models for better synthetic data generation.

1️⃣ Generate diverse but potentially lower quality synthetic data with a base model.  
2️⃣ Refine each individual data point for quality with an instruction-tuned model.  
3️⃣ Fine-tune models for downstream tasks with the final dataset.

Beyond generating training data, the idea of sampling diverse and high-quality responses from LLMs has both a large design space and broad applications -- such as creating evaluation data, generating trajectories, etc.

## 📈 Results

We evaluate **BARE** on multiple domains by fine-tuning Llama 3.1 8B Instruct and Llama 3.2 1B Instruct models for downstream tasks on **BARE**-generated data, including:

- **RAFT (Retrieval-Augmented Fine-Tuning)** → up to 18.4% improvement over SOTA data generation method.
- **GSM8K (Math Reasoning)** → 101% improvement over instruct-only data.
- **LiveCodeBench (Test Output Prediction)** → Performance comparable to SOTA models of similar size with just 1,000 samples.

We also compare **BARE** against various baselines, including base-only sampling, instruct-only sampling, and a variety of prompting methods, finding that fine-tuning with **BARE** beats baseline prompting methods and with the appropriate refiner consistently improves upon all other generation methods. In fact, solely with the addition of a small base model, **BARE**-data can consistently surpass GPT-4o only data generations.

![image](assets/comparisons.png)

For more detailed evaluations of **BARE**, check out the paper above!

## 📜 License

**BARE** is Apache 2.0 licensed, making it suitable for both academic and commercial use.

## 📧 Contact

Please feel free to reach out at aczhu@berkeley.edu & pgasawa@berkeley.edu!

## 📋 Citation

```text
@article{zhu2025bare,
  title={BARE: Leveraging Base Language Models for Few-Shot Synthetic Data Generation},
  author={Alan Zhu and Parth Asawa and Jared Quincy Davis and Lingjiao Chen and Boris Hanin and Ion Stoica and Joseph E. Gonzalez and Matei Zaharia},
  year={2025},
  journal={arXiv preprint arXiv:2502.01697},
} 
```
