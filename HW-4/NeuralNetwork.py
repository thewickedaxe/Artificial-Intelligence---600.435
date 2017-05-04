from Methods import *
import random
from random import random, seed
import numpy as np


class Node:
    """
    A class to represent nodes in the network
    """

    def __init__(self):
        pass

    inputs = []
    next_layer = []
    output = 0
    derivative = 0
    error = 0
    delta = 0
    learning_rate = 0.5

    @abstractmethod
    def activation_function(self):
        """
        abstract method that the network nodes will have to implement.
        :return: the value of the activation function applies to the input
        """
        self.output = 0
        self.derivative = 0
        self.error = 0
        self.delta = 0
        self.learning_rate = 0.05


class InputLayerNode(Node):
    """
    Class represents nodes in the input layer.
    """

    def __init__(self):
        Node.__init__(self)

    def init_val(self, value):
        """
        Changes the value of the node
        :param value: the value to assign to the node
        :return: nothing
        """
        self.output = value

    def activation_function(self):
        """
        Activation function represents the identity function.
        :return: the value of the identify
        """
        pass

    def propagate(self, index):
        """
        Propagates the output value ot the children
        :return: nothing.
        """
        for child in self.next_layer:
            if len(child.inputs) <= index:
                child.inputs.append(self.output)
            else:
                child.inputs[index] = self.output

    def back_prop(self, index):
        """
        no-op for the input nodes
        :return: nothing
        """
        pass

    def update_weight(self):
        """
        no-op for the input nodes
        :return: nothing
        """
        pass


class HiddenLayerNode(Node):
    """
    Class represents nodes in the hidden layer.
    """
    weights = []
    bias = 0

    def __init__(self, num_parents):
        """
        Assigns random values to the weights.
        """
        Node.__init__(self)
        self.weights = []
        self.bias = 0
        self.inputs = []
        for i in xrange(0, num_parents):
            self.weights.append(np.random.randn())
            #self.weights.append(0.5)
        self.bias = np.random.randn()
        #self.bias = 0.5

    def process_inputs(self):
        """
        Applies the weight an bias to the input
        :return: nothing
        """
        for i, val in enumerate(self.inputs):
            self.inputs[i] = self.inputs[i] * self.weights[i]

    def activation_function(self):
        """
        Activation function represents the sigmoid sigmoid.
        :return: nothing
        """
        self.process_inputs()
        self.output = sum(self.inputs) + self.bias
        self.output = np.power((1 + np.exp(-self.output)), -1)
        #self.output = max([self.output, 0])

    def propagate(self, index):
        """
        Propagates the output value ot the children
        :return: nothing.
        """
        self.activation_function()
        for child in self.next_layer:
            if len(child.inputs) <= index:
                child.inputs.append(self.output)
            else:
                child.inputs[index] = self.output
        self.derivative = self.output * (1.0 - self.output)

    def back_prop(self, index):
        """
        Calculates the new weights to be used
        :return: nothing
        """
        self.derivative = self.output * (1.0 - self.output)
        for child in self.next_layer:
            self.error += child.delta + child.weights[index]
        self.delta = self.error * self.derivative

    def update_weight(self):
        """
        Updates the new weight and bias
        :return: nothing
        """
        for i, weight in enumerate(self.weights):
            self.weights[i] = self.weights[i] - (self.delta * self.learning_rate * self.inputs[i])
            self.bias = self.bias + (self.delta * self.learning_rate)


class OutputLayerNode(Node):
    """
    Class represents nodes in the output layer.
    """
    weights = []
    bias = 0
    class_value = 0
    probability = 0

    def __init__(self, num_parents):
        """
        Assigns the number of parents and class value to an output node.
        :param num_parents: the number of nodes in the previous layer
        """
        Node.__init__(self)
        self.weights = []
        self.bias = 0
        self.class_value = 0
        for i in xrange(0, num_parents):
            self.weights.append(np.random.randn())
            #self.weights.append(0.5)
        self.bias = np.random.randn()
        #self.bias = 0.5
        self.probability = 0

    def process_inputs(self):
        """
        Applies the weight an bias to the input
        :return: nothing
        """
        for i, val in enumerate(self.inputs):
            self.inputs[i] = self.inputs[i] * self.weights[i]

    def activation_function(self):
        """
        Activation function represents the sigmoid sigmoid.        
        :return: nothing
        """
        self.process_inputs()
        self.output = sum(self.inputs) + self.bias
        self.output = np.power((1 + np.exp(-self.output)), -1)

    def propagate(self, index):
        """
        Performs the final activation.
        :return: nothing
        """
        self.activation_function()

    def back_prop(self, index):
        """
        Calculates the new weights to be used
        :return: nothing
        """
        self.derivative = self.output * (1.0 - self.output)
        self.error = (self.class_value - self.output)
        self.delta = self.error * self.derivative

    def update_weight(self):
        """
        Updates the new weight and bias
        :return: nothing
        """
        for i, weight in enumerate(self.weights):
            self.weights[i] = self.weights[i] - (self.delta * self.learning_rate * self.inputs[i])
            self.bias = self.bias + (self.delta * self.learning_rate)


class NeuralNetwork(Predictor):
    """
    Neural Network class
    """
    input_nodes_count = 0
    output_nodes_count = 0
    hidden_nodes_count = 0
    label_set = []
    input_layer = []
    hidden_layer = []
    hidden_layer_2 = []
    output_layer = []

    def __init__(self):
        """
        constructor for the network
        """
        self.input_nodes_count = 0
        self.output_nodes_count = 0
        self.hidden_nodes_count = 0
        self.label_set = []
        self.input_layer = []
        self.hidden_layer = []
        self.hidden_layer_2 = []
        self.output_layer = []

    def get_params(self, instances):
        """
        Analyzes the input to determine the parameters of the neural network.
        :param instances: the input data
        :return: nothing
        """
        self.input_nodes_count = len(instances[0].get_feature_vector())
        for instance in instances:
            if instance.get_label() not in self.label_set:
                self.label_set.append(instance.get_label())
        self.output_nodes_count = len(self.label_set)
        self.hidden_nodes_count = self.input_nodes_count

    def make_network(self):
        """
        Links the nodes so that they form the network structure.
        :return: nothing
        """
        for k in xrange(0, self.output_nodes_count):
            output_node = OutputLayerNode(self.hidden_nodes_count)
            self.output_layer.append(output_node)
        for j in xrange(0, self.hidden_nodes_count):
            hidden_node = HiddenLayerNode(self.hidden_nodes_count)
            hidden_node.next_layer = self.output_layer
            self.hidden_layer_2.append(hidden_node)
        for j in xrange(0, self.hidden_nodes_count):
            hidden_node = HiddenLayerNode(self.input_nodes_count)
            hidden_node.next_layer = self.hidden_layer_2
            self.hidden_layer.append(hidden_node)
        for i in xrange(0, self.input_nodes_count):
            input_node = InputLayerNode()
            input_node.next_layer = self.hidden_layer
            self.input_layer.append(input_node)

    def predict(self, instance):
        """
        Performs Prediction using a trained model.
        :param instance: The current feature vector to assign a label to
        :return: The instance with a class label assigned
        """
        for i in xrange(0, len(instance.get_feature_vector())):
            self.input_layer[i].init_val(instance.get_feature_vector()[i])
        for layer in [self.input_layer, self.hidden_layer, self.hidden_layer_2, self.output_layer]:
            self.forward_propagate(layer)
        output_vals = []
        for out_node in self.output_layer:
            output_vals.append(out_node.output)
        print self.softmax(output_vals)

    @staticmethod
    def forward_propagate(neurons):
        """
        Starts the forward propagation on the neural network.
        :return: nothing
        """
        for i, neuron in enumerate(neurons):
            neuron.propagate(i)

    @staticmethod
    def back_propagate(neurons):
        """
        Starts the forward propagation on the neural network.
        :return: nothing
        """
        for i, neuron in enumerate(neurons):
            neuron.back_prop(i)

    @staticmethod
    def update_weights(neurons):
        """
        Starts the forward propagation on the neural network.
        :return: nothing
        """
        for neuron in neurons:
            neuron.update_weight()

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.

        Rows are scores for each class. 
        Columns are predictions (samples).
        """
        max_x = max(x)
        for i, val in enumerate(x):
            x[i] = x[i] - max_x
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    def train(self, instances):
        """
        The method to train the network with
        :param instances: the instances to train with
        :return: a trained model
        """
        self.get_params(instances)
        self.make_network()
        for instance in instances * 100:
            class_index = self.label_set.index(instance.get_label())
            for o_iter, out_node in enumerate(self.output_layer):
                if o_iter == class_index:
                    out_node.class_value = 1
                else:
                    out_node.class_value = 0
            for i in xrange(0, len(instance.get_feature_vector())):
                self.input_layer[i].init_val(instance.get_feature_vector()[i])
            for layer in [self.input_layer, self.hidden_layer, self.hidden_layer_2, self.output_layer]:
                self.forward_propagate(layer)
            for layer in [self.output_layer, self.hidden_layer_2, self.hidden_layer, self.input_layer]:
                self.back_propagate(layer)
            for layer in [self.output_layer, self.hidden_layer_2, self.hidden_layer, self.input_layer]:
                self.update_weights(layer)
        for layer in [self.input_layer, self.hidden_layer, self.hidden_layer_2, self.output_layer]:
            for neuron in layer:
                print neuron.output
        out_vals = []
        for out_node in self.output_layer:
            out_vals.append(out_node.output)
        print self.softmax(out_vals)

