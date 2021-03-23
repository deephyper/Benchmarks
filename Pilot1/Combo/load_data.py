import gzip
import numpy as np

from sklearn.model_selection import train_test_split


def load_data_npz_gz(test=False):

    if test:
        fname = "testing_combo.npy.gz"
    else:
        fname = "training_combo.npy.gz"

    with gzip.GzipFile(fname, "rb") as f:
        data = np.load(f, allow_pickle=True).item()

    X, y = data["X"], data["y"]

    return X, y


def load_data():

    X, y = load_data_npz_gz()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train")
    print("Input")
    for Xi in X_train:
        print(np.shape(Xi))

    print("Output")
    print(np.shape(y_train))

    print("Valid")
    print("Input")
    for Xi in X_valid:
        print(np.shape(Xi))

    print("Output")
    print(np.shape(y_train))

    return (X_train, y_train), (X_valid, y_valid)


if __name__ == "__main__":
    load_data_npz_gz()