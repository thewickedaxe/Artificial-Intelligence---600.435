import os
import argparse
import sys
import numpy as np
import scipy.stats
from collections import Counter
from Methods import *


class NaiveBayes(Predictor):
    """
    Class implements the Predictor super class
    """

    def __init__(self):
        self.likelihoods = {}
        self.class_priors = {}

    def calc_mean(self, instances, i):
        sum = 0
        instance_count =0
        for instance in instances:
            instance_count=instance_count+1
            sum = sum + instance.get_feature_vector()[i]
        return float(sum)/instance_count

    def calc_variance(self, instances, i, mean):
        sum_standard_deviation = 0
        instance_count = 0
        for instance in instances:
            instance_count = instance_count + 1
            sum_standard_deviation =sum_standard_deviation + (instance.get_feature_vector()[i] - mean) ** 2
        return (sum_standard_deviation/ float(len(instances) - 1))

    def calc_prob(self, value, mv):
        mean, variance = mv
        gaussian = (1 / np.sqrt(2 * np.pi * variance)) * np.exp((-1 / (2 * variance)) * (value - mean) ** 2)
        return gaussian

    def train(self, instances):
        seperated_data = {}
        # Data pre-processing
        for instance in instances:
            label = instance._label.label_str
            if label not in self.class_priors:
                self.class_priors[label] = 0
                self.likelihoods[label] = {}
                seperated_data[label] = []
            seperated_data[label].append(instance)
            self.class_priors[label] += 1
            features = instance._feature_vector.feature_vec.keys()

        for label in self.class_priors:
            self.class_priors[label] = self.class_priors[label] / float(len(instances))
            for feature in features:
                mean = self.calc_mean(seperated_data[label], feature)
                variance = self.calc_variance(seperated_data[label], feature, mean)
                self.likelihoods[label][feature] = mean, variance

    def predict(self, instance):
        posteriors = {}
        features = instance._feature_vector.feature_vec.keys()
        for label in self.class_priors:
            posteriors[label] = self.class_priors[label]
            for feature in features:
                posteriors[label] *= self.calc_prob(instance._feature_vector.feature_vec[feature],
                                                    self.likelihoods[label][feature])
        if max(posteriors, key=posteriors.get) == instance._label.label_str:
            return True
        else:
            return False