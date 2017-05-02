import os
import argparse
import sys
from Methods import *


class NeuralNetwork(Predictor):
    """
    Class implements the Predictor super class
    """

    def predict(self, instance):
        pass

    def train(self, instances):
        for instance in instances:
            #print instance._feature_vector.feature_vec
            print  instance._label.label_str
