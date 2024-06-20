from source.Perceptron import Perceptron
from util.util import *
import matplotlib.pyplot as plt


def discretize_dataset_labels(dataset):
    dataset[label_name] = [1 if x == "Iris-setosa" else 0 for x in dataset[label_name]]

def get_plot(accuracy):
    max_acc, min_acc = max(accuracy), min(accuracy)
    plt.plot( accuracy)
    plt.xlabel("Num of iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs num of iterations")
    plt.axhline(y=max_acc, color='green')
    plt.text(x=0, y=max_acc, s=f'Max accuracy: {max_acc}', color='green', fontsize=8, verticalalignment='bottom')
    plt.axhline(y=min_acc, color='red')
    plt.text(x=0, y=min_acc, s=f'Min accuracy: {min_acc}', color='red', fontsize=8, verticalalignment='bottom')
    plt.show()


def test_model(dataset, perceptron):
    predictions = []
    for i in range(len(dataset[label_name])):
        obs = get_observation(dataset, i)
        p = perceptron.predict(obs)
        predictions.append(p)
        print(f"Classified {'Setosa' if obs[-1]==1 else 'Non-setosa'} {'correctly' if p==obs[-1] else 'incorrectly'} ")
    acc = perceptron.get_accuracy(dataset, predictions)
    print(f"Accuracy={acc}")

if __name__=="__main__":
    train, test = file_to_dict(r"data/iris_training.txt"), file_to_dict(r"data/iris_training.txt")
    perceptron = Perceptron(get_num_of_attributes(train), 0.2)

    discretize_dataset_labels(train)
    acc = perceptron.train(train, 30)
    # acc = perceptron.train(train)
    get_plot(acc)

    perceptron.save_model('models/perceptron_model.pkl')
