from data.formater import PandasLoader
from utils.args import Args


def main():
    args = Args()  # parse the terminal arguments

    loader = PandasLoader(args.file_path, args.format)

    data = loader.get()
    print(data)
    test = data.to_numpy().T
    print(test.shape)
    print(test)


if __name__ == "__main__":
    main()
