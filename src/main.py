import argparse

from src.models.Elman_SRN import Elman_SRN

argument_parser = argparse.ArgumentParser(
    description="Experiment with simple recursive network for statistical learning in human."
)


def main():
    print("Tac")
    srn = Elman_SRN()


if __name__ == "__main__":
    main()
