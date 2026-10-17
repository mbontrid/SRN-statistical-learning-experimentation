from enum import Enum, auto
from pathlib import Path
import pandas as pd


class Format(Enum):
    """Enum of data formats.

    Attributes:
        default: Format of data already formated for this projetc (TODO: define format).
        format_1: Format of the given excel file.
    """

    FORMAT_1 = auto()  # format of the Results_TR_24.xls file
    FORMAT_2 = (
        auto()
    )  # format of the stim_Learning_S1.xlsx file with 2 positions row selected.
    FORMAT_3 = (
        auto()
    )  # format of the stim_Learning_S1.xlsx file with 3 positions row selected.


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
        return {
            Format.FORMAT_1: self._results_tr_24_loader,
            Format.FORMAT_2: self._stim_learning_s1_2pos,
            Format.FORMAT_3: self._stim_learning_s1_3pos,
        }

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
        """Load data from the Results_TR_24.xls file.

        Returns:
            pd.DataFrame: Standardized dataframe.
        """
        # usecols = ["Trial", "Condition", "ResponseLabel", "Time", "cleaned RT"]
        usecols = ["Trial", "ResponseLabel"]

        df = pd.read_excel(self.path, sheet_name="data", usecols=usecols)
        df["Subject"] = (
            df["Trial"] == 1
        ).cumsum()  # add the count of the number of times the trial is equal to 1, and assign it to the Subject column.
        df["TrialIndex"] = df.groupby("Subject").cumcount()
        df = df.pivot(index="TrialIndex", columns="Subject", values="ResponseLabel")

        return df.astype(str)

    def _stim_learning_s1_2pos(self) -> pd.DataFrame:
        """Load data from the stim_Learning_S1.xlsx file.
        This only load the trial with L1. Meaning the ones with 2 positions.

        Returns:
            pd.DataFrame: Standardized dataframe.
        """
        usecols = ["Participant", "Sequence_id", "Block", "Sequence"]

        df = pd.read_excel(self.path, sheet_name="Sheet 1", usecols=usecols)

        df.drop(
            df[df.Sequence_id > 200].index, inplace=True
        )  # drop rows where sequence_id is greater than 200

        df.rename(
            columns={"Participant": "Subject", "Sequence": "ResponseLabel"},
            inplace=True,
        )
        df["TrialIndex"] = df.groupby("Subject").cumcount()
        df = df.pivot(index="TrialIndex", columns="Subject", values="ResponseLabel")

        return df.astype(str)

    def _stim_learning_s1_3pos(self) -> pd.DataFrame:
        """Load data from the stim_Learning_S1.xlsx file.
        This only load the trial with L2. Meaning the ones with 3 positions.

        Returns:
            pd.DataFrame: Standardized dataframe.
        """
        usecols = ["Participant", "Sequence_id", "Block", "Sequence"]

        df = pd.read_excel(self.path, sheet_name="Sheet 1", usecols=usecols)

        df.drop(
            df[df.Sequence_id < 200].index, inplace=True
        )  # drop rows where sequence_id is lower than 200

        df.rename(
            columns={"Participant": "Subject", "Sequence": "ResponseLabel"},
            inplace=True,
        )
        df["TrialIndex"] = df.groupby("Subject").cumcount()
        df = df.pivot(index="TrialIndex", columns="Subject", values="ResponseLabel")

        return df.astype(str)
