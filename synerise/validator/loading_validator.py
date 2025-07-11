import numpy as np
import logging
from typing import Optional, List, Tuple
from pathlib import Path

from synerise.validator.utils import (
    _err_msg_if_checks_fail,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class LoadingValidatorException(Exception):
    pass


class LoadingValidator:
    """
    Verifies if path exist and if embddings and client_ids can be loaded, then loads them for future use.
    """

    def __init__(self, embeddings_dir: Path):
        """
        Initializes loading validator instance.

        Args:
            embeddings_dir (Path): path to directory with embeddings
        """
        self._client_ids_pth = embeddings_dir / "client_ids.npy"
        self._embeddings_pth = embeddings_dir / "embeddings.npy"
        self._embeddings = None
        self._client_ids = None

    def _err_msg_if_client_ids_path_not_exists(self) -> Optional[str]:
        """
        Checks if client_ids path exists.

        Returns:
            Optional[str]: optional string reporting that client_ids do not exists
        """
        if not self._client_ids_pth.exists():
            return "client ids are missing"
        return None

    def _err_msg_if_client_ids_cannot_be_loaded(self) -> Optional[str]:
        """
        Checks if client ids can be loaded, then loads them.

        Returns:
            Optional[str]: optional string reporting that client ids cannot be loaded.
        """
        try:
            self._client_ids = np.load(self._client_ids_pth)
        except ValueError:
            return "client ids cannot be loaded - file does not contain ndarray"
        except EOFError:
            return "client ids cannot be loaded – file is empty"
        return None

    def _err_msg_if_embeddings_path_not_exists(self) -> Optional[str]:
        """
        Checks if the file embeddings.npy exists.

        Returns:
            Optional[str]: optional string reporting that there are no embeddings file or that there are both npy and npz files.
        """
        if not self._embeddings_pth.exists():
            return "embeddings do not exist"
        return None

    def _err_msg_if_embeddings_cannot_be_loaded(self) -> Optional[str]:
        """
        Check if embeddings can be loaded, then loads them.

        Returns:
            Optional[str]: optional string reporting that embeddings cannot be loaded.
        """
        try:
            self._embeddings = np.load(self._embeddings_pth)
        except ValueError:
            return "embeddings cannot be loaded – file does not contain ndarray"
        except EOFError:
            return "embeddings cannot be loaded - file is empty"
        return None

    def _err_msg_if_client_ids_incorrect(self) -> Optional[str]:
        """
        Runs all loading checks for client ids.

        Returns:
            Optional[str]: optional string reporting the first check that fails.
        """
        return _err_msg_if_checks_fail(
            err_msg_if_check_fail_fns=[
                self._err_msg_if_client_ids_path_not_exists,
                self._err_msg_if_client_ids_cannot_be_loaded,
            ],
            success_msg="client ids loaded",
        )

    def _err_msg_if_embeddings_incorrect(self) -> Optional[str]:
        """
        Runs all loading checks for embeddings.

        Returns:
            Optional[str]: optional string reporting the first check that fails.
        """
        return _err_msg_if_checks_fail(
            err_msg_if_check_fail_fns=[
                self._err_msg_if_embeddings_path_not_exists,
                self._err_msg_if_embeddings_cannot_be_loaded,
            ],
            success_msg="embeddings loaded",
        )

    def check_and_load(
        self,
    ) -> List[str]:
        """
        Main method that loads embeddings and client_ids.

        Returns:
            Optional[str]: optional string reporting on failure
        """
        msg_if_client_ids_incorrect = self._err_msg_if_client_ids_incorrect()
        msg_if_embeddings_incorrect = self._err_msg_if_embeddings_incorrect()
        potential_err_messages = [
            msg_if_client_ids_incorrect,
            msg_if_embeddings_incorrect,
        ]
        err_messages = [
            err_msg for err_msg in potential_err_messages if err_msg is not None
        ]
        return err_messages

    def return_client_ids_and_embeddings(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns embeddings and client_ids stored in embeddings_data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple consisting of clients_ids and embeddings
        """
        if (self._client_ids is not None) and (self._embeddings is not None):
            return self._client_ids, self._embeddings
        else:
            raise LoadingValidatorException(
                "Either client ids or embeddings are not loaded"
            )
