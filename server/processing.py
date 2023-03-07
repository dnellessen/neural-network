import sys, os

from PIL import Image
import numpy as np
from scipy import ndimage

from io import BytesIO
import base64

path = os.path.realpath(__file__)
idx = path.find("neural-network")
path = path[:idx+len("neural-network")]
sys.path.insert(0, f"{path}/nn")
from network import NeuralNetwork


nn = NeuralNetwork()
nn.load_pretrained("server_mnist_and_custom_digits_nn")



def run_through_network(img_data_base64: str) -> tuple[int, np.ndarray]:
    '''
    Returns the predictions and activations of a base 64 
    data string with the image.

    Parameters
    ----------
    img_data_base64 : str
        The base 64 data.

    Returns
    -------
    tuple[int, np.ndarray]
    '''

    inputs = get_input_data(img_data_base64)
    pred, acti = nn.prediction(inputs, activations=True)
    return pred, acti


def get_input_data(img_data_base64: str) -> np.ndarray:
    '''
    Returns the inputs for the network of a base 64 data string
    with the image.

    Parameters
    ----------
    img_data_base64 : str
        The base 64 data.

    Returns
    -------
    np.ndarray
    '''

    header, b64_img = img_data_base64.split(',')
    inputs = process_image(b64_img)
    return inputs


def process_image(b64_img: str) -> np.ndarray:
    '''
    Transforms a 28x28 base 64 image into a 784 numpy array 
    with the image's center of mass in center.

    Parameters
    ----------
    b64_img : str
        The base 64 image.

    Returns
    -------
    np.ndarray
    '''

    # resize image to (28, 28)
    img_data = base64.b64decode(b64_img)
    img_data = BytesIO(img_data)
    img = Image.open(img_data).resize((28, 28), Image.Resampling.BILINEAR)
    width, height = img.size

    # image as array + greyscale version
    inputs = np.asarray(img)
    inputs_grey = np.mean(inputs, axis=2)    # greyscale

    # calc center of mass
    com_x, com_y = ndimage.center_of_mass(inputs_grey)

    # center of image
    cx = height // 2
    cy = width  // 2

    # offset from center of mass to center of image
    ox = cx - com_x
    oy = cy - com_y

    # shift image to so the center of mass becomes the center of the image
    shifted_inputs = ndimage.shift(inputs, (ox, oy, 0))

    # image as array, normalize, and transform to greyscale
    inputs = np.asarray(shifted_inputs) / 255
    inputs = np.mean(inputs, axis=2)    # greyscale

    # flatten to (784,)
    return inputs.flatten()


def acti_as_sorted_perc(activations: np.ndarray) -> list[str]:
    '''
    Returns the activations as a sorted array in the format:
    0 - 00.00%

    Parameters
    ----------
    activations : np.ndarray

    Returns
    -------
    list[str]
    '''

    percentage = list(activations * 100)
    indexes = list(range(0, 10))

    unsorted_dict = dict(zip(indexes, percentage))
    sorted_dict = dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

    return [f"{i} :: {'{:.2f}'.format(round(p, 2))}%" for i, p in sorted_dict.items()]
