from text_classifier.model.training import train_from_config


def main():
    _, _, _, metrics, figs = train_from_config()

    print(f"Metrics:\n{metrics}")
    print(f"Figs:\n{figs}")


if __name__ == "__main__":
    main()
