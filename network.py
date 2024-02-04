"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Un módulo para implementar el algoritmo de aprendizaje de descenso de
gradiente estocástico para una red neuronal feedforward. Los gradientes 
se calculan mediante retropropagación. Tenga en cuenta que me he centrado 
en hacer que el código sea simple, fácilmente legible y fácilmente modificable. 
No está optimizado y omite muchas características deseables.

"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        
        La lista ``sizes`` contiene el número de neuronas en las
        respectivas capas de la red. Por ejemplo, si la lista fuera [2, 3, 1], 
        entonces sería una red de tres capas, donde la primera capa contendría 
        2 neuronas, la segunda capa 3 neuronas y la tercera capa 1 neurona. 
        Los sesgos y pesos de la red se inicializan aleatoriamente, utilizando 
        una distribución gaussiana con media 0 y varianza 1. Tenga en cuenta que 
        se supone que la primera capa es una capa de entrada y, por convención, 
        no estableceremos ningún sesgo para esas neuronas. ya que los sesgos solo 
        se utilizan para calcular los resultados de capas posteriores.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.

        
        Entrene la red neuronal mediante el descenso de gradiente 
        estocástico de mini lotes. ``training_data`` es una lista de tuplas 
        ``(x, y)`` que representan las entradas de entrenamiento y las salidas 
        deseadas. Los demás parámetros no opcionales se explican por sí solos. 
        Si se proporciona ``test_data``, la red se evaluará con los datos de 
        prueba después de cada época y se imprimirá el progreso parcial. 
        Esto es útil para seguir el progreso, pero ralentiza considerablemente el proceso.
        """
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        
        Actualice los pesos y sesgos de la red aplicando un descenso de 
        gradiente mediante retropropagación a un único mini lote.
        El ``mini_batch`` es una lista de tuplas ``(x, y)`` y ``eta`` 
        es la tasa de aprendizaje."""

        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
       """Tenga en cuenta que la variable l en el bucle siguiente se usa 
        de manera un poco diferente a la notación del Capítulo 2 del libro. 
        Aquí, l = 1 significa la última capa de neuronas, l = 2 es la penúltima 
        capa, y así sucesivamente. Es una renumeración del esquema del libro, 
        que se usa aquí para aprovechar el hecho de que Python puede usar índices 
        negativos en las listas. """
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        
        Devuelve el número de entradas de prueba para las cuales 
        la red neuronal genera el resultado correcto. Tenga en 
        cuenta que se supone que la salida de la red neuronal 
        es el índice de la neurona de la capa final que tenga la mayor activación.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
