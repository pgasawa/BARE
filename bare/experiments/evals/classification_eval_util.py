"""Utils for training and evaluating text classification models."""

import torch
import torch.backends.mps
import numpy as np
import random

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple

import src.logger as core
from src.tasks import ClassificationEvalTask

MODEL_CACHE_DIR = "./local_model_cache"


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }


def train_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    num_classes: int,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """BERT classification training loop

    Args:
        train_texts: List of texts to train on.
        train_labels: Corresponding list of training labels.
        val_texts: List of texts to validate on.
        val_labels: Corresponding list of validation labels.
        num_epochs: Number of epochs to run training for.
        batch_size: Size of each training batch.
        learning_rate: Learning rate.
        max_length: Maximum length of training/validation examples.

    """
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Setup device for Apple Silicon (M-series) GPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    core.logger.info(f"Using device: {device}")

    # load models
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir=MODEL_CACHE_DIR
    )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", cache_dir=MODEL_CACHE_DIR, num_labels=num_classes
    ).to(device)

    # setup datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    # train
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        metrics = evaluate_model(model, val_loader, device)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1"])

        core.logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f'Val Accuracy = {metrics["accuracy"]:.4f}, '
            f'Val F1 = {metrics["f1"]:.4f}'
        )

    return model, history


def evaluate_model(
    model: BertForSequenceClassification, data_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluates the model on the given data.

    Args:
        model: Model to evaluate.
        data_loader: DataLoader containing evaluation data.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()

    total_samples = len(data_loader.dataset)
    all_preds = np.zeros(total_samples, dtype=np.int64)
    all_labels = np.zeros(total_samples, dtype=np.int64)

    current_idx = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            batch_size = batch["input_ids"].size(0)

            # Move entire batch to device at once
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            if device.type == "mps":
                preds = preds.to("cpu")
                labels = labels.to("cpu")

            all_preds[current_idx : current_idx + batch_size] = preds.numpy()
            all_labels[current_idx : current_idx + batch_size] = labels.numpy()
            current_idx += batch_size

    # Trim arrays to actual size used
    all_preds = all_preds[:current_idx]
    all_labels = all_labels[:current_idx]

    # debug
    # unique_vals, counts = np.unique(all_preds, return_counts=True)
    # print(unique_vals)
    # print(counts)
    # unique_vals, counts = np.unique(all_labels, return_counts=True)
    # print(unique_vals)
    # print(counts)
    # mask = all_preds == all_labels
    # matched = all_preds[mask]
    # unique_vals, counts = np.unique(matched, return_counts=True)
    # print(unique_vals)
    # print(counts)

    # Compute metrics
    accuracy = np.mean(all_preds == all_labels)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def select_examples(
    texts: List[str], labels: List[int], num_samples: int, seed: int = 42
) -> Tuple[List[str], List[int]]:
    """Randomly select a subset of examples.

    Args:
        texts: List of text examples
        labels: List of corresponding labels
        num_samples: Number of examples to select
        seed: Random seed for reproducibility

    Returns:
        Tuple of (selected_texts, selected_labels)
    """
    if num_samples >= len(texts):
        return texts, labels

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    selected_indices = indices[:num_samples]
    return [texts[i] for i in selected_indices], [labels[i] for i in selected_indices]


def load_val_test_data(
    domain: ClassificationEvalTask,
    max_val_samples: int = 100,
    max_test_samples: int = 500,
):
    """Load real data for validation and testing based on task configuration.

    Args:
        task_type: Type of task to load data for.
        max_val_samples: Maximum number of validation samples to load
        max_test_samples: Maximum number of test samples to load

    Returns:
        Tuple of (val_texts, val_labels, test_texts, test_labels)
    """

    # Load validation data
    all_val_texts, all_val_labels = domain.load_val_data()
    val_texts, val_labels = select_examples(
        all_val_texts, all_val_labels, max_val_samples
    )

    # Load test data
    all_test_texts, all_test_labels = domain.load_test_data()
    test_texts, test_labels = select_examples(
        all_test_texts, all_test_labels, max_test_samples
    )

    return val_texts, val_labels, test_texts, test_labels


def train_eval(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    num_classes: int,
    num_epochs: int = 3,
) -> Dict[str, float]:
    """Main function for running the synthetic data experiment.

    Args:
        train_data: Training texts and labels.
        val_data: Validation texts and labels.
        test_data: Test texts and labels.
        num_epochs: Number of training epochs.
    """

    # Unpack data
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data

    # use available accelerator
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # Training
    model, _ = train_model(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        num_classes,
        num_epochs=num_epochs,
    )

    # Test evaluation
    test_dataset = TextClassificationDataset(
        test_texts,
        test_labels,
        BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR),
    )
    # Increased batch size for faster evaluation
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)
    metrics = evaluate_model(model, test_loader, device=device)

    return metrics
