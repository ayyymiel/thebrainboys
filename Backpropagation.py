import math as m # temporary fix to undefined variable "m" error
from random import seed
from random import random
from math import exp

# ==== Initialize Neural Network ====
def start_network(main_input: int, hidden_neur: int, output_neur: int):

    """
        This function initializes the neural network to be trained

            parameter (main_input): int
                Specifies number of inputs to feed the network
            
            parameter (hidden_neur): int
                Specifies the number of neurons in the hidden layer

            parameter (output_neur): int
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

# ---- REMOVE THIS BLOCK ----
i_count = int(input('Input Count: '))
h_count = int(input('Hidden Neuron Count: '))
o_count = int(input('Output Neuron Count: '))
# ---- REMOVE THIS BLOCK ----

get_the_network = start_network(i_count, h_count, o_count)

for layers in get_the_network:
    print(layers)

# ==== Forward Propagation ====
def activate_neur(weights, inputs):
    """
        This function generates the activation value that a neuron will have which will determine
        the neurons output. The value determines if whether or not it is going to activate it

        The higher the number, the greater the activation

        It replicates that of a linear regression function
        > neuron activation value = (neuron weight * input value) + bias

        parameter (weights): float
            Specifies the weight of each neuron from the "start_network" function
        
        parameter (inputs): float
            Specifies the value from the input to be fed to a neuron
    """
    activation = weights[-1] # assumes the generated BIAS is going to be at the end of the list of weights
    for i in range(len(weights)-1): # adds the bias to the product of the weight and input value
        activation += weights[i] * inputs[i]
    return activation

def transfer_func(activation):
    """
        This function generates the transfer function to transfer the output of the neuron to the next
        in the proceeding layer

        The activation function being used here is Sigmoid

        paremeter (activation): float
            Attains the activation values from the "activate_neur" function
    """
    sigmoid_func = 1 / 1 + exp(-activation)
    return sigmoid_func 

def forward_propagation(network, row):
    """
        This function propagates signals through the neural network layers to arrive at an output
        
        parameter (network): list
            Attains the neural network
        
        parameter (row): list
            Attains a row of data from a data set
    """
    get_inputs = row
    for layer in network: # loops through each layer generated by the network
        new_inputs = [] # makes a list of inputs for each layer looped
        for neuron in layer: # loops through each neuron in the layer
            activation = activate_neur(neuron['weights'], get_inputs)
            neuron['output'] = transfer_func(activation) # the calculated output from the transfer function will be the output of the neuron
            new_inputs.append(neuron['output']) # add the data to the new input list
        inputs = new_inputs # changes from initial inputs to the new ones
    return inputs

# ==== Back Propagation ====
def transfer_derivative(output):
    """
        This function gets the transfer derivative of the sigmoid function. The derivative of any continuous function
        is a slope. The value returned is the slope of the output value of the neuron        
        The slope represents a "gradient descent method". To have the best possible output is to have the gradient descent 
        line as flat as possible along the transfer function.
        
        Based on the slope, we can determine the error between the desired and the actual output

        (I'M NOT SURE IF THIS IS RIGHT OR NOT, I THINK IT IS, I'M NOT ENTIRELY SURE TBH)

        Links:
            (an image)  https://www.google.com/url?sa=i&url=https%3A%2F%2Fblog.clairvoyantsoft.com%2Fthe-ascent-of-gradient-descent-23356390836f&psig=AOvVaw1My1CfaW5TBOZGz3rQnYpj&ust=1629650051320000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCOCN2djGwvICFQAAAAAdAAAAABAD
            (thread explanation) https://intellipaat.com/community/17266/why-do-we-take-the-derivative-of-the-transfer-function-in-calculating-back-propagation-algorithm

        parameter (output):
            Takes the output from the output layer of the NN
    """
    t_derivative = output * (1 - output) # based on the Chain Rule
    return t_derivative


def backpropagate_error(network, desired):
    """
        This function is in charge of backpropagating the error through the network in order to see the differences in outputs between each neuron. These differences ultimately decide
        the accuracy of the model once it has been fully trained
        
        It is broken up by iterating through each layer and assigning the error differences to each neuron in them
        
        It has 2 different conditions to when assigning the error. The first being the hidden layers, the second being the output layer.
        
        (add parameters)
    """
    for layer_index in reversed(range(len(network))): # loop through each layer in the network in reverse (starting from the output layer)
        layer = network[layer_index]
        err_list = []
        if layer_index != len(network)-1: # if the current layer is NOT the output layer
            for neuron_index in range(len(layer)):
                calc_error = 0.0
                for neuron in network[layer_index + 1]: # loop through each neuron in the next layer AFTER (double-check this) the current one 
                    calc_error += neuron['weights'][neuron_index] * neuron['delta'] # calculate the error of the hidden layer neuron output
                    # above calculation is based on the weight of the neuron * the difference between expected output and actual OF THAT NEURON (RE-VISIT)
                err_list.append(calc_error)
        else:
            for neuron_index in range(len(layer)):
                neuron = layer[neuron_index]
                out_error = 0.0
                out_error += desired[neuron_index] - neuron['output'] 
                err_list.append(out_error)
        for neuron_index in range(len(layer)): # loop through each neuron in the current layer and assign delta values between each attached neuron (double-check)
            neuron = layer[neuron_index]
            neuron['delta'] = err_list[neuron_index] * transfer_derivative(neuron['output'])

# ==== START ====
random_input = [1, 0, 3, -4] # temporary data row
output_vals = forward_propagation(get_the_network, random_input)
# print(output_vals)

# FOR AMIEL, do variable checking of neuron values while running through the program to see what is being assigned to what
