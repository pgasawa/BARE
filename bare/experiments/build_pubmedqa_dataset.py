import datasets
import bm25s
import string
import random
import os
from src.tasks import DATASET_CACHE_DIR


# Preprocessing per official dataset specs
def normalize_string(s: str) -> str:

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return remove_punc(lower(s))


def process_original_dataset() -> datasets.Dataset:
    num_mismatch = 0
    dataset = datasets.load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=DATASET_CACHE_DIR,
    )

    def merge_context(context):
        merged_context = ""
        for c, la in zip(context["contexts"], context["labels"]):
            merged_context += f"{la}: {c}\n"
        return merged_context

    abstract_corpus = [merge_context(c) for c in dataset["context"]]
    formatted_corpus = [normalize_string(a) for a in abstract_corpus]
    corpus_tokens = bm25s.tokenize(formatted_corpus, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    new_contexts = []
    for i in range(len(formatted_corpus)):
        matched = False
        context = formatted_corpus[i]
        query_tokens = bm25s.tokenize(context)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
        # To return docs instead of IDs, set the `corpus=corpus` parameter.
        results, _ = retriever.retrieve(query_tokens, k=5)
        closest_contexts = []
        for j in range(results.shape[1]):
            if i == results[0][j]:
                matched = True
            closest_contexts.append(abstract_corpus[results[0][j]])
        random.shuffle(closest_contexts)
        new_contexts.append("\n".join(closest_contexts))
        if not matched:
            num_mismatch += 1
    print(num_mismatch)

    dataset = dataset.rename_column("final_decision", "answer")
    dataset = dataset.rename_column("long_answer", "reason")
    dataset = dataset.remove_columns("context")
    dataset = dataset.add_column("context", new_contexts)

    dataset = datasets.Dataset.from_list(list(dataset), features=dataset.features)

    if not os.path.exists(DATASET_CACHE_DIR + "/pub_med_qa_distractors"):
        os.makedirs(DATASET_CACHE_DIR + "/pub_med_qa_distractors")
    dataset.save_to_disk(DATASET_CACHE_DIR + "/pub_med_qa_distractors")

    return None


if __name__ == "__main__":
    process_original_dataset()
