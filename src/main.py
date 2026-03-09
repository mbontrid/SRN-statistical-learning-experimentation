from utils.formater import DataLoader
from utils.args import Args


def main():
    args = Args()  # parse the terminal arguments

    data = DataLoader()
    data.load(args.file_path, args.format)

    print(data.data)
    print(data.data.info())


if __name__ == "__main__":
    main()
