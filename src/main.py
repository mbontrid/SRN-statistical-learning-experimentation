from utils.formater import DataFrameLoader
from utils.args import Args


def main():
    args = Args()  # parse the terminal arguments

    data = DataFrameLoader()
    data.load(args.file_path, args.format)

    print(data.data)
    print(type(data.data))


if __name__ == "__main__":
    main()
