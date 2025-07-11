import logging
import argparse

from pathlib import Path

from synerise.validator.validate import (
    validate_and_load_embeddings,
)
from synerise.training_pipeline import (
    MAX_EMBEDDING_DIM,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where input embeddings are stored",
    )
    return parser


def main(params):
    """
    Main function of embedding validator.
    It calls the `validate_and_load` method with cli arguments
    for the data-dir and embeddings-dir parameters.
    """
    logger.info("validating embeddings in %s", params.embeddings_dir)
    input_dir = Path(params.data_dir) / "input"
    embeddings_dir = Path(params.embeddings_dir)
    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=embeddings_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False,
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)
