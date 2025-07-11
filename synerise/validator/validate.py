import numpy as np
import logging
from pathlib import Path
from typing import Tuple

from synerise.validator.loading_validator import (
    LoadingValidator,
)
from synerise.validator.embeddings_validator import (
    EmbeddingsValidator,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def validate_and_load_embeddings(
    input_dir: Path,
    embeddings_dir: Path,
    max_embedding_dim: int,
    disable_relevant_clients_check: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method for loading and validating embeddings.
    The method first checks if the `client_ids.npy` and `embeddings.npy`
    files exist, and imports them if they do. Otherwise returns
    a corresponding error message.
    Afterwards, the dtype, shape and other properties of the embeddings and client_ids
    is validated (see the EmbeddingsValidator class).
    If any error occures, it is reported to the user.

    Args:
        input_dir (Path): Path to the input directory (that contains `relevant_clients-npy`)
        embeddings_dir (Path): Path to the directory containing the embeddings to be loaded and
            validated.
        max_embeddings_dim (int): The maximal dimension of the embeddings.
        disable_relevant_clients_check (bool): if True, then disable checking client_ids against relevant_clients
    Returns:
        Tuple[np.ndarray, np.ndarray]: client_ids and embeddings
    """
    loading_validator = LoadingValidator(embeddings_dir=embeddings_dir)
    err_messages = loading_validator.check_and_load()
    if err_messages:
        for err_msg in err_messages:
            logger.error(err_msg)
        logger.error("Validator checks fail")
        raise SystemExit(1)
    client_ids, embeddings = loading_validator.return_client_ids_and_embeddings()
    embeddings_validator = EmbeddingsValidator(
        input_dir=input_dir,
        client_ids=client_ids,
        embeddings=embeddings,
        max_embeddings_dim=max_embedding_dim,
        disable_relevant_clients_check=disable_relevant_clients_check,
    )
    err_messages = embeddings_validator.check_embeddings()
    if err_messages:
        for err_msg in err_messages:
            logger.error(err_msg)
        logger.error("Validator checks fail")
        raise SystemExit(1)
    logger.info("Validator checks passed")

    embeddings = embeddings.astype(np.float32)

    return client_ids, embeddings
