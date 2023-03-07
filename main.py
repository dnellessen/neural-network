'''
This is a simple playground for the neural network.
You can either train the data or use a pretrained network.

Then you can either loop through the data or index through it.
In both ways you will see the plotted digit with the network's guess.

Notice: 
If you want to save a network, please do not overwrite
any existing pretrained network. 
Also, the saved network will only be usable by this file/directory.
'''


import matplotlib.pyplot as plt
import numpy as np

from nn.data import load_mnist_data, load_custom_data
from nn.functions import Cost, Activation
from nn.network import NeuralNetwork


# load training and testing data
training_data, testing_data = load_mnist_data()

# add custom data
custom_data = load_custom_data(filename="28x28_digits")
training_data = np.append(training_data, custom_data)
np.random.shuffle(training_data)
testing_data = np.append(testing_data, custom_data)
np.random.shuffle(testing_data)


nn = NeuralNetwork()

# # - Train network -
# nn.add_layer(32, num_in=784, activation=Activation.ReLU)
# nn.add_layer(10, activation=Activation.Sigmoid)
# nn.train(
#     training_data, 
#     testing_data=testing_data, 
#     epochs=10,
#     batch_size=100,
#     learn_rate=0.9,
#     verbose=True,
#     # save=True,
#     # filename="pretrained_nn"
# )

# - Loard network -
nn.load_pretrained("mnist_and_custom_digits_nn")
print('Acc: ', nn.get_accuracy(training_data), '%')

print(nn)

# - Loop through data -
for i in range(9999):
    dp = testing_data[i]
    p, a = nn.prediction(dp["inputs"], activations=True)
    print(a)

    plt.title(str(p))
    plt.imshow(dp["inputs"].reshape((28, 28)), cmap='binary')
    plt.show()

# # - Index through data -
# while True:
#     i = int(input("0 - 9999:   "))
#     dp = testing_data[i]
#     p, a = nn.prediction(dp["inputs"], activations=True)
#     print(a)

#     plt.title(str(p))
#     plt.imshow(dp["inputs"].reshape((28, 28)), cmap='binary')
#     plt.show()
