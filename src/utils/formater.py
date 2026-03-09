from enum import Enum, auto
from pathlib import Path
import pandas as pd


class Format(Enum):
    """Enum input data format.

    Attributes:
        default: Format of data already formated for this projetc (TODO: define format).
        format_1: Format of the given ecxel file.
    """

    default = auto()
    format_1 = auto()  # format of the Results_TR_24.xls file


def results_tr_24_loader(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name="data", skiprows=1)


# Associate format to corresponding loader function
format_function_dic = {Format.format_1: results_tr_24_loader}


class DataLoader:
    """Arbitrary formated data loader.
    As input format in completely arbitrary, the format has to be explicitly specified.

    Attributes:
        data: data formated for the project. Extracted from a arbitrary formated file.
    """

    def __init__(self):
        self.path: Path
        self.data: pd.DataFrame

    def load(
        self,
        file_path: Path,
        format: Format,
    ) -> pd.DataFrame:

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        self.path = file_path

        loader_function = format_function_dic.get(format)
        if loader_function is None:
            raise ValueError(f"No loader function defined for format: {format}")

        self.data = loader_function(file_path)

        return self.data
