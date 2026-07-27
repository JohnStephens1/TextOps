from text_classifier.data.model import get_encoder_train_data


def main():
    _, train_data = get_encoder_train_data()
    print(f"X_train: \n{train_data.X_train.head()}\n")
    print(f"X_test: \n{train_data.X_test.head()}\n")
    print(f"y_train: \n{train_data.y_train}\n")
    print(f"y_test: \n{train_data.y_test}\n")


if __name__ == "__main__":
    main()
