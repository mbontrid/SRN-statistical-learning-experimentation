from enum import Enum, auto
from pathlib import Path
import pandas as pd


class Format(Enum):
    """Enum of data formats.

    Attributes:
        default: Format of data already formated for this projetc (TODO: define format).
        format_1: Format of the given ecxel file.
    """

    DEFAULT = auto()
    FORMAT_1 = auto()  # format of the Results_TR_24.xls file


class DataFrameLoader:
    """Arbitrary formated data loader.
    As input format in completely arbitrary, the format has to be explicitly specified.

    Attributes:
        data: data formated for the project. Extracted from a arbitrary formated file.
    """

    def __init__(self):
        self.path: Path
        self.data: pd.DataFrame

        # Associate format to corresponding loader function
        self.FORMAT_FUNCTION_DIC = {Format.FORMAT_1: self.results_tr_24_loader}

    def load(
        self,
        file_path: Path,
        format: Format,
    ) -> pd.DataFrame:

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        self.path = file_path

        loader_function = self.FORMAT_FUNCTION_DIC.get(format)
        if loader_function is None:
            raise ValueError(f"No loader function defined for format: {format}")

        self.data = loader_function()

        return self.data

    # methods to load arbitrary foramted data. Return standardized dataframe.
    def results_tr_24_loader(self):
        data_frame = pd.read_excel(
            self.path,
            sheet_name="data",
            usecols=["Trial", "Condition", "ResponseLabel", "Time", "cleaned RT"],
        )

        return data_frame
