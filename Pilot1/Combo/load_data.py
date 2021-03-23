import gzip
import numpy as np


def load_data():
    # data_train = {"x_train": x_train_list, "y_train": y_train}

    # data_test = {"x_test_list": x_test_list, "y_test": y_test}

    with gzip.GzipFile("training_combo.npy.gz", "r") as f:
        data_train = np.load(file=f)

    X_train, y_train = data_train["x_train"], data_train["y_train"]

    # with gzip.GzipFile("testing_combo.npy.gz", "w") as f:
    # data_test = np.load(file=f)

    print("Inputs:")
    for arr in X_train:
        print(np.shape(arr))

    print("Outputs:")
    print(np.shape(y_train))

    return X_train, y_train
