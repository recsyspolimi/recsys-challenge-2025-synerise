from pathlib import Path
from synerise.data_utils.constants import PROPERTIES_FILE


class ChallengeDataDirectoryError(Exception):
    """Raised when a required directory is missing."""

    pass


class DataDir:
    """
    Container class for simplified access to subdirectories of data_dir.
    The data_dir should always conform to the structure

    data_dir/
        input/
        target/

    This class simplifies accessing files in these subdirectories by
    providing the paths to them as properties.

    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._input_dir = data_dir / "input"
        self._target_dir = data_dir / "target"
        self._properties_file = data_dir / PROPERTIES_FILE

        self._validate_data_dir()

    @property
    def data_dir(self) -> Path:
        """
        Path to data_dir
        """
        return self._data_dir

    @property
    def input_dir(self) -> Path:
        """
        Path to data_dir/input_dir
        """
        return self._input_dir

    @property
    def target_dir(self) -> Path:
        """
        Path to data_dir/target_dir.
        """
        return self._target_dir

    @property
    def properties_file(self) -> Path:
        """
        Path to product properties file.
        """
        return self._properties_file

    def _validate_data_dir(self) -> None:
        """
        Method for validating that the structure of the provided data_dir
        conforms to the descripting outlined in the competition description.
        """
        if not self._data_dir.exists():
            raise ChallengeDataDirectoryError(
                f"Directory '{self._data_dir}' does not exist"
            )

        if not self._input_dir.exists():
            raise ChallengeDataDirectoryError(
                f"The 'input' subdirectory in '{self._data_dir}' is missing; directory with competition data must contain an 'input' directory"
            )

        if not self._target_dir.exists():
            raise ChallengeDataDirectoryError(
                f"The 'target' subdirectory in '{self._data_dir}' is missing; directory with competition data must contain a 'target' directory"
            )

        if not self._properties_file.exists():
            raise ChallengeDataDirectoryError(
                f"The {PROPERTIES_FILE} file missing in {self._data_dir}"
            )
