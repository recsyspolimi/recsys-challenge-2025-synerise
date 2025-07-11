import numpy as np
import logging

from typing import List, Callable, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


@dataclass
class EmbeddingsData:
    """
    Stores embeddings and client_ids.
    """

    embeddings: Optional[np.ndarray] = None
    client_ids: Optional[np.ndarray] = None


def _err_msg_if_checks_fail(
    err_msg_if_check_fail_fns: List[Callable[[], Optional[str]]],
    success_msg: str,
) -> Optional[str]:
    """
    Function that runs cascade of checks and returns report of the first check that fails. If all checks are succesfull, then success message is logged.

    Args:
        err_msg_if_check_fail_fns (List[Callable[[], Optional[str]]]): list of check functions to run in cascade
        success_msg (str): success message logged if all checks are passed

    Returns:
        Optional[str]: optional string reporting first check in cascade that fail.
    """
    for pass_msg_if_check_fail in err_msg_if_check_fail_fns:
        msg_if_check_fail = pass_msg_if_check_fail()
        if msg_if_check_fail:
            return msg_if_check_fail
    else:
        logger.info(success_msg)
        return None
