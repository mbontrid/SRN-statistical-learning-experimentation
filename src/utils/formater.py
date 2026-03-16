from enum import Enum, auto
from pathlib import Path
import pandas as pd
import numpy as np


class Format(Enum):
    """Enum of data formats.

    Attributes:
        default: Format of data already formated for this projetc (TODO: define format).
        format_1: Format of the given ecxel file.
    """

    DEFAULT = auto()
    FORMAT_1 = auto()  # format of the Results_TR_24.xls file


class LoaderToNumpy:
    """Arbitrary formated file data loader.
    As input format in completely arbitrary, the format has to be explicitly specified.

    Attributes:
        data: data formated for the project. Extracted from a arbitrary formated file.
    """

    def __init__(self, path: Path, format: Format, nrow: int | None = None):
        self._path: Path = path
        self.format: Format = format
        self.nrow: int | None = nrow  # Number of rows per yield.

        # Associate format to corresponding loader function
        self._FORMAT_FUNCTION_DIC = {Format.FORMAT_1: self.results_tr_24_loader}

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        if not p.exists():
            raise FileNotFoundError(f"File {self.path} not found.")

        self._path = p

    def get(self, path: Path | None = None, nrow: int | None = None) -> np.ndarray:
        self.nrow = nrow
        if self.nrow is not None:
            raise NotImplementedError(
                "nrow (to yield array) argument is not implemented yet."
            )

        self.path = path if path is not None else self.path
        return self._load_ndarray()

    def _load_dataframe(
        self,
    ) -> pd.DataFrame:

        loader_function = self._FORMAT_FUNCTION_DIC.get(self.format)
        if loader_function is None:
            raise ValueError(f"No loader function defined for format: {format}")

        self.data = loader_function()

        return self.data

    def _load_ndarray(self) -> np.ndarray:
        data_frame = self._load_dataframe()

        data_ndarray = data_frame.to_numpy()

        return data_ndarray

    # methods to load arbitrary foramted data. Return standardized dataframe.
    def results_tr_24_loader(self):
        data_frame = pd.read_excel(
            self.path,
            sheet_name="data",
            usecols=["Trial", "Condition", "ResponseLabel", "Time", "cleaned RT"],
        )

        return data_frame
