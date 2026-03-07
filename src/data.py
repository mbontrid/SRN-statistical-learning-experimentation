import pandas as pd
from pathlib import Path


class data_formater:
    """Arbitrary formated data loader.

    Attributes:
        data: data formated for the project. Extracted from a arbitrary formated file.
    """

    def __init__(self, file_path: Path):

        file_extension = file_path.suffix

        self.data = None

        if file_extension == "xls":
            self.data = self.load_xls(file_path)

    @staticmethod
    def load_xls(file_path: Path):
        pass


class data_sequence:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def read_excel(self, file_path):
        df = pd.read_excel(file_path)
        print(df)
