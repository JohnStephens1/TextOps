from text_classifier.model.models import RandomForestModel
from text_classifier.model.training import train_qm


def main():
    my_model = RandomForestModel()
    _, _, _, metrics, figs = train_qm(my_model)

    print(f"Metrics:\n{metrics}")
    print(f"Figs:\n{figs}")


if __name__ == "__main__":
    main()
