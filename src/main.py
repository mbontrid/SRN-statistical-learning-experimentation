from data.formater import LoaderToNumpy
from utils.args import Args


def main():
    args = Args()  # parse the terminal arguments

    loader = LoaderToNumpy(args.file_path, args.format)

    data = loader.get()
    data = data.flatten()
    print(data)
    print(data.dtype)
    print(data.shape)
    print(type(data[0]))


if __name__ == "__main__":
    main()
