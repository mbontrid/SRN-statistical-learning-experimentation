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


class PandasLoader:
    """Arbitrary formated file data loader.
    As input format in completely arbitrary, the format has to be explicitly specified.

    Attributes:
        data: data formated for the project. Extracted from a arbitrary formated file.
    """

    def __init__(self, path: Path, format: Format, nrow: int | None = None):
        self._path: Path = Path(path).resolve()
        self.format: Format = format
        self.nrow: int | None = nrow  # Number of rows per yield.
        if self.nrow is not None:
            raise NotImplementedError("nrow yiel not yet implemented")

        # Associate format to corresponding loader function

    @property
    def format_function(self) -> dict:
        return {Format.FORMAT_1: self._results_tr_24_loader}

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        if not p.exists():
            raise FileNotFoundError(f"File {p} not found.")

        self._path = p

    def get(self, path: Path | None = None) -> pd.DataFrame:

        self.path = path if path is not None else self.path
        return self._load()

    def _load(
        self,
    ) -> pd.DataFrame:

        loader_function = self.format_function.get(self.format)
        if loader_function is None:
            raise ValueError(f"No loader function defined for format: {format}")

        self.data = loader_function()

        return self.data

    # ---------------------------------------------------------------
    # methods to load arbitrary foramted data. Return standardized dataframe.
    # ---------------------------------------------------------------

    def _results_tr_24_loader(self) -> pd.DataFrame:
        # usecols = ["Trial", "Condition", "ResponseLabel", "Time", "cleaned RT"]
        usecols = ["Trial", "ResponseLabel"]

        df = pd.read_excel(
            self.path,
            sheet_name="data",
            usecols=usecols,
        )
        df["Subject"] = (df["Trial"] == 1).cumsum()
        df["TrialIndex"] = df.groupby("Subject").cumcount()
        df = df.pivot(index="TrialIndex", columns="Subject", values="ResponseLabel")

        return df
