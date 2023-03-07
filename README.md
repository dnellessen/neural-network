[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

# Neural Network

Neural network that can be trained to recognize digits, which is its main application for this project.

https://user-images.githubusercontent.com/108455731/223578056-f217e601-ee9a-4c6a-89a2-732a32038689.mov

<br/>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
        <a href="#about-the-project">About The Project</a>
        <ul>
            <li>
                <a href="#math">Math</a>
                <ul>
                    <li><a href="#forward-propagation">Forward Propagation</a></li>
                    <li><a href="#backward-propagation">Backward Propagation</a></li>
                    <li><a href="#matrix-multiplication">Matrix Multiplication</a></li>
                    <li><a href="#readings">Readings</a></li>
                </ul>
            </li>
            <li><a href="#mnist-data">MNIST Data</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## About The Project

### Math
<img src="https://www.mdpi.com/BDCC/BDCC-02-00016/article_deploy/html/images/BDCC-02-00016-g001.png" width="70%">


### Forward Propagation
The weighted sum of a layer is calculated by the dot product of the weights and the inputs, plus the biases:
$$Z_L = W_L \cdot A_{L-1} + b_L$$

To get the activations, the weighted sum is passed through an activation function like Sigmoid or ReLU:
$$A_L = \sigma(Z_L)$$



Sigmoid: $$f(x) = \frac{1}{1+e^{-x}}$$
ReLU: $$f(x) = max(0, x)$$

### Backward Propagation 
Gradient Descent:

Calculate the derivative of the cost function with respect to the weights and biases using the chain rule:

Output Layer:
$$X_L = \frac{\partial A_L}{\partial Z_L} \cdot \frac{\partial C_0}{\partial A_L}$$
$$\frac{\partial C_0}{\partial W_L} = \frac{\partial Z_L}{\partial W_L} \cdot X_L = A_{L-1} \cdot X_L$$
$$\frac{\partial C_0}{\partial b_L} = \frac{\partial Z_L}{\partial b_L} \cdot X_L = 1 \cdot X_L$$

Hidden Layers:
$$X_{L-1} = \frac{\partial A_{L-1}}{\partial Z_{L-1}} \cdot \frac{\partial Z_L}{\partial A_{L-1}} \cdot X_{L}$$
$$\frac{\partial C_0}{\partial W_{L-1}} = \frac{\partial Z_{L-1}}{\partial W_{L-1}} \cdot X_{L-1} = A_{L-2} \cdot X_{L-1}$$
$$\frac{\partial C_0}{\partial b_{L-1}} = \frac{\partial Z_{L-1}}{\partial b_{L-1}} \cdot X_{L-1} = 1 \cdot X_{L-1}$$

Then these gradients are added to the layers' weights and bias gradients to later subtract them from the weights and biases (after multiplying them with a learning rate):
$$W_L = \alpha \cdot G_{W, L}$$
$$b_L = \alpha \cdot G_{b, L}$$

### Matrix Multiplication
Dot Product

$$ \begin{pmatrix} 
w_{1,1}&w_{1,2}&\dots &w_{1,n}\\ 
w_{2,1}&w_{2,2}&\dots &w_{2,n}\\ 
\vdots &\vdots &\ddots &\vdots \\ 
w_{m,1}&w_{m,2}& \dots  &w_{m,n} 
\end{pmatrix}
\cdot 
\begin{pmatrix}
v_1 \\
v_2 \\ 
\vdots \\ 
v_n 
\end{pmatrix} = 
\begin{pmatrix} 
\sum_{k=1}^n w_{k,1}v_{k}\\ 
\sum_{k=1}^n w_{k,2}v_{k}\\ 
\vdots \\ 
\sum_{k=1}^n w_{k,m}v_{k}\\ 
\end{pmatrix} $$


Outer Product

$$ \begin{pmatrix}
a_1 \\
a_2 \\
\vdots \\
a_m
\end{pmatrix}
\otimes
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix} =
\begin{pmatrix}
a_{1}x_{1}&a_{1}x_{2}&\dots &a_{1}x_{n}\\
a_{2}x_{1}&a_{2}x_{2}&\dots &a_{2}x_{n}\\
\vdots &\vdots &\ddots &\vdots \\
a_{m}x_{1}&a_{m}x_{2}&\dots &a_{m}x_{n}
\end{pmatrix} $$

Inner Product

$$ \begin{pmatrix}
w_{1,1}&w_{1,2}&\dots &w_{1,n}\\
w_{2,1}&w_{2,2}&\dots &w_{2,n}\\
\vdots &\vdots &\ddots &\vdots \\
w_{m,1}&w_{m,2}&\dots &w_{m,n}
\end{pmatrix}
\cdot
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix} =
\begin{pmatrix}
\sum_{k=1}^n w_{1,k}x_{k}\\
\sum_{k=1}^n w_{2,k}x_{k}\\
\vdots \\
\sum_{k=1}^n w_{m,k}x_{k}\\
\end{pmatrix} $$

### Readings
* [3blue1brown - Neural Networks](https://www.3blue1brown.com/topics/neural-networks)


### MNIST Data
![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) provides a training set of 60,000 examples, and a test set of 10,000 examples.
The data is formatted and saved in files so the network can work with it.

Since the MNIST dataset mainly has an 'American' style of digits (like the 1 and the 9), I've added a custom dataset of 1,000 examples with a 'European' style.
Also because the MNIST dataset images are centered in such way that their center of mass of the pixels is at the center of the image, the same processing had to be implemented to the custom dataset, and is implemented by the server.


## Getting Started

### Installation
```sh
$ git clone https://github.com/dnellessen/neural-network.git
```

### Prerequisites
* Flask
* matplotlib
* mnist
* numpy
* Pillow
* scipy
* torchvision
    ```bash
    $ pip install -r requirements.txt
    ```


## Usage
In the `main.py` file you can load or train a network and see the plotted digit with the network's prediction. Read the docstring in the file for more information.

To write digits yourself and see what digit the network thinks it is, simply run the Flask server.
```bash
$ pwd
.../neural-network

$ python3 server/app.py
```


## License
Distributed under the MIT License. See `LICENSE` for more information.
