import pickle
import math
import util.util

label_name = "label"
def get_dataset_size(dataset):
    return len(dataset[label_name])
def get_observation(dataset, index):
    return [dataset[x][index] for x in dataset]
def get_k_smallest(l, k):
    return sorted(l)[:k]
def get_indices(l, target_list):
    indices=[]
    for i in range(len(l)):
        for j in range(len(target_list)):
            if l[i] == target_list[j]:
                if not indices.__contains__(j):
                    indices.append(j)

    return indices
class KNN:
    def __init__(self, train_set):
        self.train_set = train_set

    def get_difference(self, observation, data):  # returns a single float value representing the difference of one observation to another
        differences = [math.pow(observation[x] - data[x], 2) for x in range(len(observation) - 1)]
        return math.sqrt(sum(differences))

    def get_closest_indices(self, observation, k):
        differences = []
        for i in range(get_dataset_size(self.train_set)):
            x = get_observation(self.train_set, i)
            differences.append(self.get_difference(observation, x))
        smallest = get_k_smallest(differences, k)  # smallest differences
        indices = get_indices(smallest, differences)  # indicies of smallest differences
        return indices, smallest

    def classify(self, indices, train_set, prnt=False):
        categories = {x: 0 for x in set(train_set[label_name])}
        for i in indices:
            categories[train_set[label_name][i]] += 1
        if prnt: print(categories)
        return max(categories, key=categories.get)

    def predict(self, observation, k):
        indices, _ = self.get_closest_indices(observation, k)
        return self.classify(indices, self.train_set)


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model( file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

# if __name__ == '__main__':
#     train = util.util.file_to_dict(r"../data/iris_training.txt")
#     model = KNN(train)
#     model2 = KNN.load_model(r'../models/knn_model.pkl')
#     print(model.train_set)
#     print(model2.train_set)
#     print(model.predict([ 7.0   , 	 3.2    ,	 4.7    ,	 1.4    ,	 "Iris-versicolor" ], 3))
