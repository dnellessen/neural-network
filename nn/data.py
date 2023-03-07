import torchvision.datasets
from mnist import MNIST

import numpy as np
import os


def load_custom_data(filename: str) -> np.ndarray:
    '''
    Loads custom dataset.

    The data must consist of an array with dictionaries consisting of the
    inputs (np.ndarray), the label (int), and the expected outputs (np.ndarray).

    Parameters
    ----------
    filename : str
        The name of the file (.npy) in the data/custom/ directory.

    Returns
    -------
    np.ndarray
    '''

    custom_data = __read_file(f"{__PATH_DATA}/custom/{filename}.npy")
    return custom_data


def load_mnist_data() -> tuple:
    '''
    Loads the data from from the MNIST dataset.

    Returns both the training data and the testings data individually.
    Each consists of an array with dictionaries consisting of the
    inputs (np.ndarray), the label (int), and the expected outputs (np.ndarray).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
    '''

    training_npy_exists = os.path.exists(f"{__PATH_MNIST}/training.npy")
    testing_npy_exists  = os.path.exists(f"{__PATH_MNIST}/testing.npy")

    if not (training_npy_exists and testing_npy_exists):
        training_data, testing_data = __process_data()
        __write_to_file(training_data, f"{__PATH_MNIST}/training.npy")
        __write_to_file(testing_data, f"{__PATH_MNIST}/testing.npy")

    training_data = __read_file(f"{__PATH_MNIST}/training.npy")
    testing_data = __read_file(f"{__PATH_MNIST}/testing.npy")
    return training_data, testing_data


def __process_data():
    '''
    Processes the training/testing data to the dicts.
    
    Pixels are transformed from range 0-255 to range 0-1.
    '''

    train_imgs, train_lbls = __DATASET.load_training()
    test_imgs, test_lbls = __DATASET.load_testing()

    training_data = [
        {
            "inputs": np.array((train_imgs[i])) / 255, 
            "label": train_lbls[i],
            "exp_outputs": __one_hot(train_lbls[i])
        } for i in range(len(train_imgs))]


    testing_data = [
        {
            "inputs": np.array(test_imgs[i]) / 255, 
            "label": test_lbls[i],
            "exp_outputs": __one_hot(test_lbls[i])
        } for i in range(len(test_imgs))]

    return np.array(training_data), np.array(testing_data)


def __one_hot(label: int):
    ''' 
    Returns the expected outputs.

    A one-hot is a group where only a single bit is high and all the others are low. 

    Parameters
    ----------
    label : int
        The label (becoming the index).

    Returns
    -------
    np.ndarray
        The expected outputs.
    '''
    
    expected_outputs = np.zeros(10)
    expected_outputs[label] = 1
    return expected_outputs


def __write_to_file(data: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        np.save(f, data)


def __read_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)

    return data


def __download_raw():
    torchvision.datasets.MNIST(
        root=__PATH_DATA,
        download=True
    )


def __get_main_path() -> str:
    path = os.path.realpath(__file__)
    idx = path.find("neural-network")
    return path[:idx+len("neural-network")]


__PATH_MAIN = __get_main_path()
__PATH_DATA = f"{__PATH_MAIN}/data"
__PATH_MNIST = f"{__PATH_DATA}/MNIST"

# download the MNSIT dataset it it doesn't exist'
if not os.path.exists(f"{__PATH_MNIST}/raw"):
    __download_raw()

__DATASET = MNIST(f"{__PATH_MNIST}/raw")
