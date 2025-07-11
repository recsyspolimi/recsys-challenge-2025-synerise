import numpy as np
from pathlib import Path

def align_embedding(client_ids, embedding):
    if np.array_equal(client_ids, sorted(client_ids)):
        return client_ids, embedding
    else:
        sorted_indices = np.argsort(client_ids)
        sorted_client_ids = client_ids[sorted_indices]
        sorted_embedding = embedding[sorted_indices]

        return sorted_client_ids, sorted_embedding


def load_clients_and_embeddings(dataset_path):
    if type(dataset_path) == str:
        dataset_path = Path(dataset_path)

    embeddings = np.load(dataset_path / "embeddings.npy")
    client_ids = np.load(dataset_path / "client_ids.npy")

    return align_embedding(client_ids, embeddings)

