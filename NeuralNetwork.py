import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import pandas as pd
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

def softmax(x):
    max = np.max(x)
    exp_x = np.exp(x - max)
    return exp_x / exp_x.sum()

def softmax_derivative(x):
    max = np.max(x)
    exp_x = np.exp(x - max)
    return exp_x/np.sum(exp_x, axis=0) * (1 - exp_x/np.sum(exp_x, axis=0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def loss(predicted_output, desired_output):
    return -np.sum(desired_output * np.log(predicted_output))

class NeuralNetwork():
    def __init__(self, inputLayerNeuronsNumber, hiddenLayerNeuronsNumber, outputLayerNeuronsNumber, learning_rate):
        self.inputLayerNeuronsNumber = inputLayerNeuronsNumber
        self.hiddenLayerNeuronsNumber = hiddenLayerNeuronsNumber
        self.outputLayerNeuronsNumber = outputLayerNeuronsNumber
        self.learning_rate = learning_rate

        self.hidden_weights = np.random.randn(hiddenLayerNeuronsNumber, inputLayerNeuronsNumber) * np.sqrt(2 / inputLayerNeuronsNumber)
        self. hidden_weights2 = np.random.randn(hiddenLayerNeuronsNumber, hiddenLayerNeuronsNumber) * np.sqrt(2 / hiddenLayerNeuronsNumber)
        self.hidden_bias = np.zeros([hiddenLayerNeuronsNumber, 1])
        self.hidden_bias2 = np.zeros([hiddenLayerNeuronsNumber, 1])

        self.output_weights = np.random.randn(outputLayerNeuronsNumber, hiddenLayerNeuronsNumber)
        self.output_bias = np.zeros([outputLayerNeuronsNumber, 1])
        self.loss = []

    def train(self, inputs, desired_output):
        #FORWARD PROPAGATION
        hidden_layer_in = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_layer_out = sigmoid(hidden_layer_in)

        hidden_layer_in_2 = np.dot(self.hidden_weights2, hidden_layer_out) + self.hidden_bias2
        hidden_layer_out_2 = sigmoid(hidden_layer_in_2)

        output_layer_in = np.dot(self.output_weights, hidden_layer_out_2) + self.output_bias
        predicted_output = softmax(output_layer_in)

        #CALCULATING ERROR
        error = desired_output - predicted_output
        d_predicted_output = error * softmax_derivative(predicted_output)

        error_hidden_layer_2 = d_predicted_output.T.dot(self.output_weights)
        d_hidden_layer_2 = error_hidden_layer_2.T * sigmoid_derivative(hidden_layer_out_2)

        error_hidden_layer = d_hidden_layer_2.T.dot(self.hidden_weights2)
        d_hidden_layer = error_hidden_layer.T * sigmoid_derivative(hidden_layer_out)

        #BACKWARD PROPAGATION
        self.output_weights += hidden_layer_out_2.dot(d_predicted_output.T).T * self.learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate

        self.hidden_weights2 += hidden_layer_out.dot(d_hidden_layer_2.T).T * self.learning_rate
        self.hidden_bias2 += np.sum(d_hidden_layer_2, axis=0, keepdims=True) * self.learning_rate

        self.hidden_weights += inputs.dot(d_hidden_layer.T).T * self.learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
        self.loss.append(loss(predicted_output, desired_output))

    def predict(self, inputs):
        hidden_layer_in = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_layer_out = sigmoid(hidden_layer_in)
        hidden_layer_in_2 = np.dot(self.hidden_weights2, hidden_layer_out) + self.hidden_bias2
        hidden_layer_out_2 = sigmoid(hidden_layer_in_2)
        output_layer_in = np.dot(self.output_weights, hidden_layer_out_2) + self.output_bias
        return softmax(output_layer_in)

#LOADING MNIST DATASET -------------------------------------------------------------
x1, y1 = loadlocal_mnist(images_path='./data/train-images.idx3-ubyte', labels_path='./data/train-labels.idx1-ubyte')
num_train1 = 50000
num_test1 = 10000
x_train1 = x1[:num_train1, :]/255
y_train1 = np.zeros((num_train1, 10))
y_train1[np.arange(0, num_train1), y1[:num_train1]] = 1
x_test1 = x1[num_train1, :]/255
y_test1 = np.zeros((num_test1, 10))
y_test1[np.arange(0, num_test1), y1[y1.size - num_test1:]] == 1

#LOADING  SELF MADE DATASET---------------------------------------------------------
dataset = np.load('dataset.npy')

with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    y = list(reader)

for i in range(len(y[0])):
    y[0][i] = int(y[0][i])

#SPLIT = 4 : 1 ----------------------------------------------------------

num_of_data = 11789

num_train2 = int(round(4/5 * num_of_data))
num_test2 = num_of_data - num_train2

x2 = dataset.reshape(num_of_data, 784)
y2 = np.array(y[0])

#PERSEBARAN DATASET ----------------------------------------------------------

#COMBINING SELF MADE AND MNIST -------------------------------------------------
num_train = num_train1 + num_train2
num_test = num_test1 + num_test2

x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)

randomize = np.arange(len(x))

np.random.shuffle(randomize)

x = x[randomize]
y = y[randomize]

x_train = x[:num_train, :]/255
x_test = x[num_train:, :]/255

y_train_label = [x for x in y[:num_train]]
y_test_label = [x for x in y[num_train:]]
# plt.hist([y_train_label, y_test_label], edgecolor='black', stacked=True, bins=19, label=['Training Set', 'Test Set'])
# plt.show()

y_train = np.zeros((num_train, 10))
y_train[np.arange(0, num_train), y_train_label] = 1
y_test = np.zeros((num_test, 10))
y_test[np.arange(0, num_test), y_test_label] = 1

# fig, ax = plt.subplots(1, 1, figsize=(5,5), tight_layout=True, edgecolor='black')
# ax.hist(y)

starting_lr = 0.01

nn = NeuralNetwork(784, 250, 10, starting_lr)

accuracy_list = []
epochs = 60
for epoch in range(epochs):

    acc_file = open('data/top_acc.txt', 'r+')
    highestAccuracy = float(acc_file.read())
    print("===================================================================")

    print("Epoch: " + str(epoch + 1))
    print("Highest accuracy so far : " + str(highestAccuracy))
    for i in range(len(x_train)):
        inputs = np.array(x_train[i, :]).reshape(-1, 1)
        outputs = np.array(y_train[i, :]).reshape(-1, 1)
        nn.train(inputs, outputs)

    prediction_list = []
    for i in range(len(x_test)):
        inputs = np.array(x_test[i].reshape(-1, 1))
        prediction_list.append(nn.predict(inputs))

    correct_counter = 0
    output_list = np.array([])
    print(prediction_list[0])
    for i in range(len(prediction_list)):
        out_index = np.where(prediction_list[i] == np.amax(prediction_list[i]))[0][0]
        output_list = np.append(output_list, out_index)

        if y_test[i][out_index] == 1:
            correct_counter += 1

    accuracy = correct_counter / num_test

    print("Accuracy is : ", accuracy * 100, " %")
    accuracy_list.append(accuracy)

    if accuracy > highestAccuracy:
        np.save('data/weights/hiddenWeight.npy', nn.hidden_weights)
        np.save('data/weights/hiddenWeight2.npy', nn.hidden_weights2)
        np.save('data/weights/outputWeight.npy', nn.output_weights)
        np.save('data/weights/hiddenBias.npy', nn.hidden_bias)
        np.save('data/weights/hiddenBias2.npy', nn.hidden_bias2)
        np.save('data/weights/outputBias.npy', nn.output_bias)
        print("Saved new weights with accuracy ", accuracy * 100, " %")
        acc_file.seek(0)
        acc_file.write(str(accuracy))
        acc_file.truncate()

    ground_truth = np.array([])
    for i in range(num_test):
        ground_truth = np.append(ground_truth, np.where(y_test[i] == 1)[0][0])

    cm = confusion_matrix(ground_truth, output_list)
    cm_df = pd.DataFrame(cm, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                         index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(cm_df)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(1,epochs+1), accuracy_list)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Neural Network using softmax activation function on the output layer')
plt.show()