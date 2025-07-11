from enum import Enum


class TaskNotSupportedError(Exception):
    pass


class ValidTasks(Enum):
    pass


class ChurnTasks(ValidTasks):
    CHURN = "churn"


class PropensityTasks(ValidTasks):
    PROPENSITY_CATEGORY = "propensity_category"
    PROPENSITY_SKU = "propensity_sku"


def parse_task(task_name: str) -> ValidTasks:
    """
    Method for parsing the input arg to command line arg --task into a ValidTask

    Args:
        task_name (str): name of the task, i.e. string representation of the task as read from the --task argument.

    Returns:
        ValidTasks
    """
    for task_type in ValidTasks.__subclasses__():
        try:
            return task_type(task_name)
        except ValueError:
            continue
    raise TaskNotSupportedError(f"Task: {task_name} is not supported")


def get_propensity_column(task: PropensityTasks) -> str:
    """
    Method for obtaining the string representation of the column of a
    propensity task. For example:

    PropensityTasks.PropensitySku -> sku
    """
    return task.value.split("_")[-1]
