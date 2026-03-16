import argparse
from pathlib import Path

from utils.formater import Format


class Args:
    def __init__(self):

        parser = argparse.ArgumentParser(
            description="Experimentation (playground) of simple recursive network in the domain of human statistical learning."
        )

        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            default=Path("./data/input/Results_TR_24.xls"),
            help="Path to the file to be loaded.",
        )

        parser.add_argument(
            "--format",
            "-f",
            type=Format,
            default=Format.FORMAT_1,
            choices=list(Format),
            help="Format of the given file. This specify the file extension and the arbitrary formating.",
        )

        args = parser.parse_args()

        self.file_path: Path = args.input
        self.format: Format = args.format
