import numpy as np
from Methods import *


class NaiveBayes(Predictor):
    def __init__(self):
        self.class_probabilities = {}
        self.probs = {}

    @staticmethod
    def moment_1(instances, i):
        summation = 0
        instance_count = 0
        for instance in instances:
            instance_count = instance_count + 1
            summation = summation + instance.get_feature_vector()[i]
        return float(summation) / instance_count

    @staticmethod
    def moment_2(instances, i, mean):
        sum_standard_deviation = 0
        instance_count = 0
        for instance in instances:
            instance_count = instance_count + 1
            sum_standard_deviation = sum_standard_deviation + (instance.get_feature_vector()[i] - mean) ** 2
        return sum_standard_deviation / float(len(instances) - 1)

    def calc_feature_wise_probabilities(self, instance, label, pre_prob):
        for i in instance.get_feature_vector().keys():
            pre_prob[label] *= self.calc_prob(instance.get_feature_vector()[i],
                                              self.probs[label][i])

    @staticmethod
    def calc_prob(value, mv):
        mju_1, mju_2 = mv
        gaussian = (1 / np.sqrt(2 * np.pi * mju_2)) * np.exp((-1 / (2 * mju_2)) * (value - mju_1) ** 2)
        return gaussian

    def separate_data(self, instances):
        separated_data = {}
        for instance in instances:
            label = instance.get_label()
            self.make_separation_decision(label, separated_data)
            separated_data[label].append(instance)
            self.class_probabilities[label] += 1
            instance_count = instance.get_feature_vector().keys()
        return separated_data, instance_count

    def make_separation_decision(self, label, separated_data):
        if label not in self.class_probabilities:
            self.class_probabilities[label] = 0
            self.probs[label] = {}
            separated_data[label] = []

    def train(self, instances):
        separated_data, attributes_length = self.separate_data(instances)
        for label in self.class_probabilities:
            self.class_probabilities[label] = self.class_probabilities[label] / float(len(instances))
            self.calc_moments(attributes_length, label, separated_data)

    def calc_moments(self, attributes_length, label, separated_data):
        for i in attributes_length:
            mju_1 = self.moment_1(separated_data[label], i)
            mju_2 = self.moment_2(separated_data[label], i, mju_1)
            self.probs[label][i] = mju_1, mju_2

    def predict(self, instance, cur_index=0):
        pre_prob = {}
        for label in self.class_probabilities:
            pre_prob[label] = self.class_probabilities[label]
            self.calc_feature_wise_probabilities(instance, label, pre_prob)
        return max(pre_prob, key=pre_prob.get)
