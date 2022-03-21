import numpy as np
from sklearn.model_selection import train_test_split

def get_train_valid(data_path):
    # load data
    loaded = np.load(data_path)
    data = loaded['data']
    labels = loaded['labels']
    print("Data Shape: ", data.shape, labels.shape)

    # normalize data
    mean = data.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
    std = data.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
    data_normal = (data-mean)/std

    # add dimension of 1
    data_normal = np.expand_dims(data_normal, axis=3)
    print("Data shape after adding dimension: ", data_normal.shape)

    # Split the data for training and validation
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data_normal, labels, test_size=0.2, shuffle=True)

    return data_train, data_valid, labels_train, labels_valid

