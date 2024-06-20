import pickle

from util.util import *


class Perceptron:

    def __init__(self, num_inputs, learning_rate):
        self.weights = self.get_weights(num_inputs)
        self.bias = 0.1
        self.learning_rate = learning_rate

    def get_weights(self, amount):
        return [get_rand_nonzero() for _ in range(amount)]

    def activation(self, x):
        return 1 if x >= 0 else 0

    def get_delta(self, true_label, prediction):
        return true_label - prediction

    def dot_product(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"length of vectors does not match ({len(x)} vs {len(y)}): {x} and {y}")
        return sum([x[i] * y[i] for i in range(len(x))])

    def predict(self, observation):
        net = self.dot_product(observation[:-1], self.weights)
        return self.activation(net - self.bias)

    def little_train(self, observation):
        p = self.predict(observation)
        if p != observation[-1]:
            delta = self.get_delta(observation[-1], p)
            self.correct_bias(delta)
            self.correct_weights(observation, delta)
        return p

    def correct_weights(self, observation, delta):
        new_weights = [self.weights[i] + (delta * observation[i] * self.learning_rate) for i in
                       range(len(self.weights))]
        self.weights = new_weights

    def correct_bias(self, delta):
        new_bias = self.bias + (delta * self.learning_rate)
        self.bias = new_bias

    def get_accuracy(self, dataset, predictions):
        correct = 0
        for i in range(len(predictions)):
            if dataset[label_name][i] == predictions[i]:
                correct += 1
        return correct / len(predictions)

    def train(self, dataset, epochs=0):
        accuracy = 0.0
        previous_accuracy = []
        predictions = []
        if epochs == 0:
            previous_accuracy = []
            i = 0
            counter = 1
            while (accuracy < 1):
                observation = get_observation(dataset, i)
                predictions.append(self.little_train(observation))
                if i == get_dataset_size(dataset) - 1:  # last element
                    i = 0
                    accuracy = self.get_accuracy(dataset, predictions)
                    previous_accuracy.append(accuracy)
                    self.print_state(counter, accuracy)
                    counter += 1
                    predictions.clear()
                    if counter > 50 and len(set(previous_accuracy[-10:])) == 1:
                        return previous_accuracy
                    # if counter % 50 == 0:
                    #     input("pause")
                i += 1


        else:
            for i in range(epochs):
                for j in range(get_dataset_size(dataset)):
                    observation = get_observation(dataset, j)
                    predictions.append(self.little_train(observation))
                accuracy = self.get_accuracy(dataset, predictions)
                previous_accuracy.append(accuracy)
                self.print_state(i, accuracy)
                predictions.clear()
            return previous_accuracy

    def print_state(self, counter, accuracy):
        print(f"Iteration {counter}: accuracy={accuracy}, weights={self.weights}, bias={self.bias}")

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)