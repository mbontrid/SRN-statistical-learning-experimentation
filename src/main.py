from utils.formater import LoaderToNumpy
from utils.args import Args
from models.Elman_SRN.torch_Elman_SRN import ElmanSRN


def main():
    args = Args()  # parse the terminal arguments

    loader = LoaderToNumpy(args.file_path, args.format)

    data = loader.get()
    data = data.flatten()
    print(data)
    print(data.dtype)
    print(data.shape)
    print(type(data[0]))

    srn = ElmanSRN()

    srn.load_numpy(data)


if __name__ == "__main__":
    main()
