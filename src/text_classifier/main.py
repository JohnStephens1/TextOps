from text_classifier.model.models import RandomForestModel
from text_classifier.model.training import train_qm


def main():
    my_model = RandomForestModel()
    _, _, _, metrics, figs = train_qm(my_model)

    print(metrics)
    print(figs)


if __name__ == "__main__":
    main()
