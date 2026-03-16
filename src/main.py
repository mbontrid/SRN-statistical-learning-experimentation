from utils.formater import LoaderToNumpy
from utils.args import Args


def main():
    args = Args()  # parse the terminal arguments

    loader = LoaderToNumpy(args.file_path, args.format)

    data = loader.get()
    print(data)
    print(type(data))


if __name__ == "__main__":
    main()
