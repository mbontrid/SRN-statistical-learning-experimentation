import argparse
from data import Format, RawData
from pathlib import Path

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


def main():
    data = RawData()
    data.load(args.input, args.format)
    print(data.data)


if __name__ == "__main__":
    main()
