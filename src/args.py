import argparse
from data import Format
from pathlib import Path


class Args:
    def __init__(self):

        parser = argparse.ArgumentParser(
            description="Experimentation (playground) of simple recursive network in the domain of human statistical learning."
        )

        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            default=Path("./data/Results_TR_24.xls"),
            help="Path to the file to be loaded.",
        )

        parser.add_argument(
            "--format",
            "-f",
            type=Format,
            default=Format.format_1,
            help="Format of the given file. This specify the file extension and the arbitrary formating.",
        )

        args = parser.parse_args()

        self.file_path: Path = args.input
        self.format: Format = args.format
