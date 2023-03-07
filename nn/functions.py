import numpy as np


class Cost:
    class MSE:
        '''
        Mean Squared Error

        y_hat : predicted outputs
        y : expected outputs

        Methods
        -------
        f(y_hat, y)
            Function.
        fi(y_hat, y)
            Derivative.
        '''

        @staticmethod
        def f(y_hat, y):
            return (y_hat - y) ** 2

        @staticmethod
        def fi(y_hat, y):
            return 2 * (y_hat - y)


class Activation:
    class Sigmoid:
        '''
        Methods
        -------
        f(y_hat, y)
            Function.
        fi(y_hat, y)
            Derivative.
        '''

        @staticmethod
        def f(x):
            return 1 / (1 + np.exp(-x))

        @staticmethod
        def fi(x):
            f_x = 1 / (1 + np.exp(-x))
            return f_x * (1 - f_x)

    class ReLU:
        '''
        Methods
        -------
        f(y_hat, y)
            Function.
        fi(y_hat, y)
            Derivative.
        '''
        
        @staticmethod
        def f(x):
            return np.maximum(0, x)

        @staticmethod
        def fi(x):
            return np.greater(x, 0).astype(int)
