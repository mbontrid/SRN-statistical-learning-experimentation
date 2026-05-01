import argparse
from pathlib import Path
from collections.abc import Iterable

try:
    from data.formater import Format
except ModuleNotFoundError:
    from src.data.formater import Format


def get_args(input: Path | None = None, format: Format | None = None) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    return {
        "input": input if input is not None else project_root / "data/input/Results_TR_24.xls",
        "format": format if format is not None else Format.FORMAT_1,
    }


class Args:
    def __init__(self, args: Iterable[str] | None = None):

        default_args = get_args()

        parser = argparse.ArgumentParser(
            description="Experimentation (playground) of simple recursive network in the domain of human statistical learning."
        )

        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            default=default_args["input"],
            help="Path to the file to be loaded.",
        )

        parser.add_argument(
            "--format",
            "-f",
            type=Format,
            default=default_args["format"],
            choices=list(Format),
            help="Format of the given file. This specify the file extension and the arbitrary formating.",
        )

        parsed = parser.parse_args(args=args)

        self.file_path: Path = parsed.input
        self.format: Format = parsed.format
