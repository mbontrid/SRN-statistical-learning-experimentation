from data import DataLoader
from args import Args


def main():
    args = Args()  # parse the terminal arguments

    data = DataLoader()
    data.load(args.file_path, args.format)

    print(data.data)


if __name__ == "__main__":
    main()
