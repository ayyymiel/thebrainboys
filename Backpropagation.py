import math as m # temporary fix to undefined variable "m" error
from random import seed
from random import random

def start_network(main_input: int, hidden_neur: int, output_neur: int):

    """
        This function initializes the neural network to be trained

            parameter (main_input):
                Specifies number of inputs to feed the network
            
            parameter (hidden_neur):
                Specifies the number of neurons in the hidden layer

            parameter (output_neur):
                Specifies the number of neurons in the output layer

            Code reference: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    """

    # network list
    network = []

    # PLEASE NOTE: +1 is added to include the bias to the neuron
    """
        Below generates the hidden layer(s)
        > It is created by randomly generating the weights and biases for the input channels for
            the number of neurons in the hidden layer
    """
    hidden_layer = [{'weights':[random() for i in range(main_input + 1)]} for i in range(hidden_neur)]
    network.append(hidden_layer)

    """
        Below generates the output layer(s)
        > It is created by randomly generating the weights and biases for the channels fed to the 
            number of neurons in the output layer
    """
    output_layer = [{'weights':[random() for i in range(hidden_neur + 1)]} for i in range(output_neur)]
    network.append(output_layer)

    return network # return the layer information

i_count = int(input('Input Count: '))
h_count = int(input('Hidden Neuron Count: '))
o_count = int(input('Output Neuron Count: '))

seed(1) # start random numbers at 1 
get_the_network = start_network(i_count, h_count, o_count)

for layers in get_the_network:
    print(layers)
