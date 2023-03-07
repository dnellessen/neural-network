import numpy as np
import os

try:
    from .functions import Cost, Activation
except ImportError:
    from functions import Cost, Activation


class NeuralNetwork:
    def __init__(self, cost_func=Cost.MSE):
        '''
        A neural network class that can train a network using gradient descent and make predictions.

        Parameters
        ----------
        cost_func : class
            The cost function of the network.

        Attributes
        ----------
        func : class
            The cost function of the network.

        Methods
        -------
        add_layer(size, num_in, activation=Activation.Sigmoid)
            Adds a layer to the network.
        load_pretrained(filename)
            Load pretrained network.
        get_accuracy(data)
            Gets the accuracy of the network for the given data (in %).
        prediction(inputs, activations=False)
            Gets the networks's prediction for the given inputs.
        train(training_data, epochs=10, batch_size=100, learn_rate=0.25, save=False, filename="trained_nn", testing_data=np.array([]), verbose=True)
            Trains the neural network using gradient descent.
        '''

        self.func = cost_func
        self._layers = []

    def __repr__(self) -> str:
        cost_func = self.func.__name__

        if len(self._layers) == 0:
            return f"<NeuralNetwork - X | X | {cost_func} >"

        shape = [str(self._layers[0].num_in)]
        acti_func = []
        for layer in self._layers:
            shape.append(str(layer.num_out))
            acti_func.append(layer.func.__name__)

        shape = 'x'.join(shape)
        acti_func = ', '.join(acti_func)

        return f"<NeuralNetwork - {shape} | {acti_func} | {cost_func} >"

    def add_layer(self, size: int, num_in: int = None, activation=Activation.Sigmoid) -> None:
        '''
        Adds a layer to the network.

        Parameters
        ----------
        size : int
            The size of the layer.
        num_in : int (optional)
            The size of the layer before - only for first layer (default is None).
        activation : class (optional)
            The activation function of the layer (default is Activation.Sigmoid).
        '''

        num_in = self._layers[-1].num_out if num_in is None else num_in
        self._layers.append(Layer(num_in, size, activation))

    def load_pretrained(self, filename: str = "trained_nn") -> None:
        '''
        Load pretrained network.

        Parameters
        ----------
        filename : str (optional)
            The name of the file (.npy) in the /nn/pretrained directory (default is 'trained_nn').
        '''

        path = os.path.realpath(__file__)
        idx = path.find("neural-network")
        path = f"{path[:idx+len('neural-network')]}/nn/pretrained"

        with open(f"{path}/{filename}.npy", 'rb') as f:
            self._layers = np.load(f, allow_pickle=True)

    def get_accuracy(self, data: np.ndarray) -> float:
        '''
        Gets the accuracy of the network for the given data (in %).

        Parameters
        ----------
        data : np.ndarray
            An array with all the data.

        Returns
        -------
        float
            The percentage.
        '''

        predictions = [(self.prediction(dp["inputs"]), dp["label"]) for dp in data]
        d = np.sum(int(pred == label) for (pred, label) in predictions) / len(data)
        return round(d * 100, 3)

    def prediction(self, inputs: np.ndarray, activations: bool = False) -> tuple[int, np.ndarray] | int:
        '''
        Gets the networks's prediction for the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs of a data map.
        activations : bool (optional)
            If True the output layer's activations are returned (default is False).

        Returns
        -------
        tuple[int, np.ndarray] | int
            The index of the highest output node (and its activations).
        '''

        y_hat = self._forward(inputs)

        if activations:
            return np.argmax(y_hat), self._layers[-1].A
        return np.argmax(y_hat)

    def train(self, training_data: np.ndarray, epochs: int = 10, batch_size: int = 100, learn_rate: float = 0.25, 
              save: bool = False, filename: str = "trained_nn", testing_data: np.ndarray = np.array([]), verbose: bool = True) -> None:
        '''
        Trains the neural network using gradient descent.

        Parameters
        ----------
        training_data : np.ndarray
            All the training data maps.
        epochs : int (optional)
            The number of epochs (iterations) to run all the 
            data through the network (default is 10).
        batch_size : int (optional)
            The batch size each batch should be (default is 100).
        learn_rate : float (optional)
            The learn rate (default is 0.25).
        save : bool (optional)
            Save the network in .npy file (default is False).
        filename : str (optional)
            The filename to save the network - only if save is True (default is False).
        verbose : bool (optional)
            Flag to print the accuracy after every epoch (default is "trained_nn").
        testing_data : np.ndarray (optional)
            The testing data to calculate the accuracy - only if 
            verbose is true (default is 10).
        '''
        
        np.random.shuffle(training_data)
        batches = np.split(
            training_data,
            np.arange(
                batch_size,
                len(training_data),
                batch_size
            )
        )

        verb_and_training = verbose and testing_data.any()
        if verb_and_training:
            print(f"Acc: {self.get_accuracy(testing_data)}%")

        for epoch in range(epochs):
            for batch in batches:
                self._backprop(batch, learn_rate)
            if verb_and_training:
                acc = self.get_accuracy(testing_data)
                print(f"Epoch: {epoch+1}/{epochs}", end=' ')
                print(f"Acc: {acc}%")

        if save and filename:
            if verb_and_training:
                print(f"Saving natwork in /nn/pretrained/{filename}.npy")
            self._save_network(filename)

    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Forward propagation through the entire network.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs of a data map.

        Returns
        -------
        np.ndarray
            The output layer's activations.
        '''

        for layer in self._layers:
            inputs = layer.activations(inputs)
        return inputs

    def _backprop(self, batch: np.ndarray, learn_rate: float) -> None:
        '''
        Backpropagation through the entire network.

        (1) Update all gradients for every batch.
        
        (2) Apply all gradients.
        
        (3) Clear all gradients.

        Parameters
        ----------
        batch : np.ndarray
            The batch with all the data maps.
        learn_rate : float
            The learn rate.
        '''

        for dp in batch:
            self._update_all_gradients(dp)
        self._apply_all_gradients(learn_rate / len(batch))
        self._clear_all_gradients()

    def _update_all_gradients(self, data: dict) -> None:
        '''
        Updates all gradients of each layer.

        Parameters
        ----------
        data : dict
        '''

        self._forward(data["inputs"])

        outer_layer = self._layers[-1]
        group = outer_layer.func.fi(
            outer_layer.Z) * self.func.fi(outer_layer.A, data["exp_outputs"])
        outer_layer.update_gradients(group)

        for layer_i in range(len(self._layers) - 2, -1, -1):
            hidden_layer = self._layers[layer_i]
            prev_layer = self._layers[layer_i + 1]
            group = hidden_layer.func.fi(
                hidden_layer.Z) * np.inner(prev_layer.W, group)
            hidden_layer.update_gradients(group)

    def _apply_all_gradients(self, learn_rate: float) -> None:
        '''
        Applies all gradients of each layer.

        Parameters
        ----------
        learn_rate : float
        '''

        for layer in self._layers:
            layer.apply_gradients(learn_rate)

    def _clear_all_gradients(self) -> None:
        ''' Clears all gradients of each layer. '''

        for layer in self._layers:
            layer.clear_gradients()

    def _save_network(self, filename: str = "trained_nn") -> None:
        '''
        Save pretrained network.

        Parameters
        ----------
        filename : str (optional)
            The name of the file (.npy) in the /nn/pretrained directory (default is 'trained_nn').
        '''

        path = os.path.realpath(__file__)
        idx = path.find("neural-network")
        path = f"{path[:idx+len('neural-network')]}/nn/pretrained"

        with open(f"{path}/{filename}.npy", 'wb') as f:
            np.save(f, self._layers)


class Layer:
    def __init__(self, num_in: int, num_out: int, activation_func):
        '''
        A class that handles a single layer of a neural network.

        Parameters
        ----------
        num_in : int
            The number in inputs.
        num_out : int
            The number of outputs.
        activation_func : class
            The activation function of the layer.

        Attributes
        ----------
        num_in : int
            The number in inputs.
        num_out : int
            The number of outputs.
        func : class
            The activation function of the layer.
        W : np.ndarray
            The layer's weights.
        b : np.ndarray
            The layer's biases.
        W_gradients : np.ndarray
            The layer's weight gradients.
        b_gradients : np.ndarray
            The layer's bias gradients.
        A_prev : float
            The layer's given inputs (previous activations).
        A : float
            The layer's activations.
        Z : float
            The layer's weighted sum 
            (before the passing it through the activation function).

        Methods
        -------
        weighted_sum(inputs)
            Calculates the weighted sum of given inputs.
        activations(inputs)
            Calculates the activations of given inputs.
        apply_gradients(learn_rate)
            Applies the gradients to the weights and biases.
        update_gradients(group)
            Updates the weight and bias gradients.
        clear_gradients()
            Resets the weight and bias gradients.
        '''

        self.num_in = num_in
        self.num_out = num_out

        self.func = activation_func

        self.W = np.random.uniform(-0.5, 0.5, size=(num_in, num_out))
        self.b = np.random.uniform(-0.5, 0.5, size=(num_out,))

        self.W_gradients = np.zeros((num_in, num_out))
        self.b_gradients = np.zeros((num_out,))

        self.A_prev = 0.
        self.A = 0.
        self.Z = 0.

    def weighted_sum(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Calculates the weighted sum of given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs of a data map.

        Returns
        -------
        np.ndarray
            The weighted sum.
        '''
        
        self.A_prev = inputs
        self.Z = self.W.T @ inputs + self.b
        return self.Z

    def activations(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Calculates the activations of given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs of a data map.

        Returns
        -------
        np.ndarray
            The activations.
        '''

        self.weighted_sum(inputs)
        self.A = self.func.f(self.Z)
        return self.A

    def apply_gradients(self, learn_rate: float) -> None:
        '''
        Applies the gradients to the weights and biases.

        Parameters
        ----------
        learn_rate : float
        '''

        self.W -= learn_rate * self.W_gradients
        self.b -= learn_rate * self.b_gradients

    def update_gradients(self, group: np.ndarray) -> None:
        '''
        Updates the weight and bias gradients.

        Parameters
        ----------
        group : np.ndarray
            The calculated group that's the same for the weight and bias gradients.
        '''

        self.W_gradients += np.outer(self.A_prev, group)
        self.b_gradients += 1 * group

    def clear_gradients(self) -> None:
        ''' Resets the weight and bias gradients. '''
        self.W_gradients = np.zeros((self.num_in, self.num_out))
        self.b_gradients = np.zeros((self.num_out,))
