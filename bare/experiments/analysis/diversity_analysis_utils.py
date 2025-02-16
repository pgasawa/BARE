"""Diversity Analysis via cosine similarity."""

import os
import torch
import numpy as np

from diskcache import Cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict

from src.models.llm_engine import EmbeddingEngine
from src.tasks import DataGenerationTask

# Cache initialization
_cache_dir = Path(os.path.dirname(__file__)) / ".cache" / "embeddings"
_disk_cache = Cache(str(_cache_dir), eviction_policy="none")
_embedding_cache = {}  # In-memory cache

# Device initialization
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_embedding(trace: str) -> tuple[np.ndarray, float]:
    # Check in-memory cache first
    if trace in _embedding_cache:
        return _embedding_cache[trace], 0.0

    # Check disk cache
    if trace in _disk_cache:
        embedding = _disk_cache[trace]
        _embedding_cache[trace] = embedding  # Load into memory
        return embedding, 0.0

    # Compute new embedding
    embedding_engine = EmbeddingEngine(model_name="text-embedding-3-small")
    response = embedding_engine.get_embedding(trace)
    embedding = np.array(response.embedding)

    # Cache the result both in memory and on disk
    _embedding_cache[trace] = embedding
    _disk_cache[trace] = embedding

    return embedding, response.cost


def analyze_diversity(
    task: DataGenerationTask, data_points: List[Dict[str, str]]
) -> tuple[float, float]:
    """Takes in a list of reasoning traces and returns a tuple: (num_unique_traces, average_trace_similarity, cost)."""
    global _embedding_cache

    # too few traces
    if len(data_points) <= 1:
        return 0, 0

    # format for embedding
    formatted_data = [task.format_for_embedding(d) for d in data_points]

    # running totals
    cost = 0

    # Pre-compute all embeddings up front
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(compute_embedding, trace) for trace in formatted_data
        ]

        with tqdm(total=len(formatted_data), desc="Computing embeddings") as pbar:
            for future in as_completed(futures):
                embedding, indiv_cost = future.result()
                _embedding_cache[formatted_data[futures.index(future)]] = embedding
                cost += indiv_cost
                pbar.update(1)

    # Convert all embeddings to a single tensor for batch processing
    embeddings_list = [_embedding_cache[trace] for trace in formatted_data]

    # Convert list to single numpy array first
    embeddings_array = np.array(embeddings_list)

    # Convert to float32 and move to GPU
    embeddings = torch.from_numpy(embeddings_array).float().to(device)

    # Normalize the embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute all pairwise similarities at once using matrix multiplication
    similarity_matrix = torch.mm(embeddings, embeddings.t())

    # Get upper triangle indices (excluding diagonal)
    indices = torch.triu_indices(len(formatted_data), len(formatted_data), offset=1)
    similarities = similarity_matrix[indices[0], indices[1]]

    # Convert to CPU for numpy operations
    similarities = similarities.cpu()
    similarity_matrix = similarity_matrix.cpu()

    # Compute average similarity
    average_trace_similarity = similarities.mean().item()

    return average_trace_similarity, cost
