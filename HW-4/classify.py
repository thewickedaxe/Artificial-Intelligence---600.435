import argparse
import os
import pickle

from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
from NeuralNetwork import NeuralNetwork
from Methods import ClassificationLabel, FeatureVector, Instance


def load_data(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue

            # Divide the line into features and label.
            split_line = line.split(",")
            label_string = split_line[0]

            label = ClassificationLabel(label_string)
            feature_vector = FeatureVector()

            index = 0
            for item in split_line[1:]:
                value = float(item)

                feature_vector.add(index, value)
                index += 1

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances


def get_args():
    parser = argparse.ArgumentParser(
        description="This allows you to specify the arguments you want for classification.")

    parser.add_argument("--data", type=str, required=True,
                        help="The data files you want to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="Filename specifying where to save or load model.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    args = parser.parse_args()
    check_args(args)

    return args


def predict(predictor, instances):
    total_count = 0
    correct_count = 0
    precision_base = {}
    precision_correct = {}
    recall_base = {}
    recall_correct = {}
    cur_index = -1
    i=0
    if predictor.__class__.__name__ == "DecisionTree":
        labels = predictor.predict(instances)
    else:
        for instance in instances:
            cur_index += 1
            label = predictor.predict(instance)
            if label == instance._label.label_str:
                correct_count = correct_count + 1
            total_count = total_count + 1
            if label not in precision_correct:
                precision_correct[label] = 0
                precision_base[label] = 0
            if instance._label.label_str == label:
                precision_correct[instance._label.label_str] += 1
            precision_base[label] += 1
            if instance._label.label_str not in recall_correct:
                recall_correct[instance._label.label_str] = 0
                recall_base[instance._label.label_str] = 0
            if instance._label.label_str == label:
                recall_correct[instance._label.label_str] += 1
            recall_base[instance._label.label_str] += 1
            #print(str(label))

def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm must be specified in mode \"train\"")
    else:
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(instances, algorithm):
    """
    This is where you tell classify.py what algorithm to use for training
    The actual code for training should be in the Predictor subclasses
    For example, if you have a subclass DecisionTree in Methods.py
    You could say
    if algorithm == "decision_tree":
        predictor = DecisionTree()
    """
    if algorithm == "decision_tree":
        predictor = DecisionTree()
    elif algorithm == "neural_network":
        predictor = NeuralNetwork()
    else:
        predictor = NaiveBayes()
    predictor.train(instances)
    return predictor


def main():
    args = get_args()
    if args.mode.lower() == "train":
        # Load training data.
        instances = load_data(args.data)

        # Train
        predictor = train(instances, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif args.mode.lower() == "test":
        # Load the test data.
        instances = load_data(args.data)

        predictor = None
        # Load model
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")

        predict(predictor, instances)
    else:
        raise Exception("Unrecognized mode.")


if __name__ == "__main__":
    main()
