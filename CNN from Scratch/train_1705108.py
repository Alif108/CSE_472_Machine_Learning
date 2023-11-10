import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import pandas as pd
import math
import os
from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

#################################### Preprocessing ####################################
def one_hot_encode(labels, n_classes=10):
    """
    One-hot encodes a numpy array of labels
    params:
        labels: numpy array of labels
        n_classes: number of classes
    """
    n_labels = len(labels)
    one_hot_encode = np.zeros((n_labels, n_classes))
    
    for i in range(n_labels):
        one_hot_encode[i, labels[i]] = 1

    return one_hot_encode

def load_train_dataset(X_train_path, Y_train_path, image_size=(28, 28), start_idx = 0, samples_to_load=None):
    """
    Loads the training dataset
    params:
        X_train_path: path to the folder containing the training images
        Y_train_path: path to the csv file containing the labels
        image_size: size of the images
        start_idx: index of the first sample to load
        samples_to_load: number of samples to load
    """

    # Load the labels
    Y_df = pd.read_csv(Y_train_path)
    num_samples = Y_df.shape[0]

    if samples_to_load is None:
        if start_idx == 0:
            samples_to_load = num_samples
        else:
            samples_to_load = num_samples - start_idx

    if start_idx + samples_to_load > num_samples:
        samples_to_load = num_samples
    
    Y_df = Y_df[start_idx : start_idx + samples_to_load]                # Select "samples_to_loads" samples starting from "start_idx"
    Y_train = np.array(Y_df["digit"])
    filenames = np.array(Y_df["filename"])
    Y_train = Y_train.reshape(samples_to_load, 1)
    Y_train = one_hot_encode(Y_train)

    # Load the images
    # extension = "*.png"
    # filenames = glob.glob(X_train_path + extension)
    # filenames = os.listdir(X_train_path)
    # filenames = filenames[start_idx : start_idx + samples_to_load]      # Select "samples_to_loads" samples starting from "start_idx"

    # print(filenames[:5])
    # print(Y_train[:5])

    num_channels = 3
    input_image_len = len(np.array(Image.open(X_train_path + filenames[0])).shape)
    if input_image_len == 2:
        num_channels = 1

    X_train = []
    for filename in filenames:
        image = Image.open(X_train_path + filename)
        image = np.array(image.resize(image_size))
        X_train.append(image)

    X_train = np.array(X_train)
    X_train = (255 - X_train) / 255
    X_train = X_train.reshape(samples_to_load, image_size[0], image_size[1], num_channels)

    print("Number of files loaded: ", len(filenames))

    return X_train, Y_train


#################################### Loss Prime ####################################
def categorical_crossentropy_prime(y_true, y_pred):
    """
    Calculates the derivative of the categorical cross-entropy loss function
    params:
        y_true: true labels
        y_pred: predicted labels
    """
    return y_pred - y_true


#################################### Layers ####################################
class Layer:
    def __init__(self):
        self.X = None
        self.Y = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

    def clear_cache(self):
        self.X = None
        self.Y = None

class FC(Layer):
    def __init__(self, output_dim):
        self.Y_dim = output_dim
        self.W = None
        self.b = None

    def forward(self, input):
        self.X = input
        num_pixels = input.shape[1]

        if self.W is None:
            self.W = np.random.randn(self.Y_dim, num_pixels) * np.sqrt(2 / num_pixels)
        if self.b is None:
            self.b = np.zeros((1, self.Y_dim))

        self.Y = np.dot(self.X, self.W.T) + self.b
        return self.Y

    def backward(self, dY, learning_rate):
        dW = np.dot(self.X.T, dY)
        db = np.sum(dY, axis=0, keepdims=True)
        dX = np.dot(dY, self.W)

        self.W -= learning_rate * dW.T
        self.b -= learning_rate * db
        
        return dX


class Flatten(Layer):
    def __init__(self):
        self.X_shape = None

    def forward(self, input):
        self.X_shape = input.shape
        reshaped_input = input.reshape(input.shape[0], -1)
        return reshaped_input

    def backward(self, dY, learning_rate):
        reshaped_dY = dY.reshape(self.X_shape)
        return reshaped_dY

class Softmax(Layer):
    def __init__(self):
        self.X = None

    def forward(self, input):
        self.X = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        output = exps / np.sum(exps, axis=1, keepdims=True)
        return output

    def backward(self, dY, learning_rate):
        return dY

class ReLU(Layer):
    def __init__(self):
        self.X = None

    def forward(self, input):
        self.X = input
        return np.maximum(0, input)

    def backward(self, dY, learning_rate):
        dX = dY.copy()
        dX[self.X <= 0] = 0
        return dX

def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

class Conv2D(Layer):
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None

    def forward(self, input):
        self.X = input
        num_samples, input_height, input_width, num_channels = input.shape
        
        out_h = int((input_height - self.filter_size + 2 * self.padding) / self.stride) + 1
        out_w = int((input_width - self.filter_size + 2 * self.padding) / self.stride) + 1

        if self.W is None:
            self.W = np.random.randn(self.num_filters, self.filter_size, self.filter_size, num_channels) * np.sqrt(2 / (self.filter_size * self.filter_size * num_channels))
        if self.b is None:
            self.b = np.zeros(self.num_filters)

        self.X_pad = np.pad(self.X, ((0,), (self.padding,), (self.padding,), (0,)), 'constant')
        X_trans = np.transpose(self.X_pad, (0, 3, 1, 2))
        W_trans = np.transpose(self.W, (0, 3, 1, 2))
    
        self.windows = getWindows(X_trans, (num_channels, num_channels, out_h, out_w), self.filter_size, self.padding, self.stride)

        Y_trans = np.einsum('bihwkl,oikl->bohw', self.windows, W_trans)

        # add bias to kernels
        Y_trans += self.b[None, :, None, None]

        self.Y = np.transpose(Y_trans, (0, 2, 3, 1))
        
        return self.Y

    def backward(self, dY, learning_rate):
        num_samples, input_height, input_width, num_channels = self.X.shape

        dY_trans = np.transpose(dY, (0, 3, 1, 2))
        W_trans = np.transpose(self.W, (0, 3, 1, 2))

        padding = self.filter_size - 1 if self.padding == 0 else self.padding

        x_pad_trans_shape = np.transpose(self.X_pad, (0, 3, 1, 2)).shape
        dout_windows = getWindows(dY_trans, x_pad_trans_shape, self.filter_size, padding=padding, stride=1, dilate=self.stride - 1)
        rot_kern = np.rot90(W_trans, 2, axes=(2, 3))
        
        db = np.sum(dY_trans, axis=(0, 2, 3))
        dw_trans = np.einsum('bihwkl,bohw->oikl', self.windows, dY_trans)
        dx_pad_trans = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        dw = np.transpose(dw_trans, (0, 2, 3, 1))
        dx_pad = np.transpose(dx_pad_trans, (0, 2, 3, 1))

        self.W -= learning_rate * dw
        self.b -= learning_rate * db

        return dx_pad[:, self.padding:self.padding + input_height, self.padding:self.padding + input_width, :]

class MaxPool(Layer):
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride

    def forward(self, input):
        self.X = input
        num_samples, input_height, input_width, num_channels = input.shape
        output_height = int((input_height - self.filter_dim) / self.stride) + 1
        output_width = int((input_width - self.filter_dim) / self.stride) + 1
        self.Y = np.zeros((num_samples, output_height, output_width, num_channels))

        for n in range(num_samples):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(num_channels):
                        self.Y[n, i, j, c] = np.max(input[n, i * self.stride:i * self.stride + self.filter_dim, j * self.stride:j * self.stride + self.filter_dim, c])

        return self.Y


    def backward(self, dY, learning_rate):
        num_samples, input_height, input_width, num_channels = self.X.shape
        num_samples, output_height, output_width, num_filters = dY.shape

        dX = np.zeros((num_samples, input_height, input_width, num_channels))
        
        for n in range(num_samples):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(num_channels):
                        i_t, j_t = np.where(np.max(self.X[n, i * self.stride:i * self.stride + self.filter_dim, j * self.stride:j * self.stride + self.filter_dim, c]) == self.X[n, i * self.stride : i * self.stride + self.filter_dim, j * self.stride:j * self.stride + self.filter_dim, c])
                        i_t, j_t = i_t[0], j_t[0]
                        dX[n, i * self.stride + i_t, j * self.stride + j_t, c] = dY[n, i, j, c]
        
        return dX

# class for a cnn model
class CNN:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, X, y_true, learning_rate):
        y_pred = self.forward(X)
        loss = self.loss(y_true, y_pred)
        d_loss = self.loss_prime(y_true, y_pred)
        self.backward(d_loss, learning_rate)
        return loss

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_loss, learning_rate):
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, learning_rate)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y = np.argmax(y, axis=1)

        accuracy = accuracy_score(y, y_pred)
        score = f1_score(y, y_pred, average='macro')
        confusion = confusion_matrix(y, y_pred)

        return accuracy, score, confusion

    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()


def main():

    x_train_a_path = "Dataset/training-a/"
    y_train_a_path = "Dataset/training-a.csv"
    x_train_b_path = "Dataset/training-b/"
    y_train_b_path = "Dataset/training-b.csv"
    x_train_c_path = "Dataset/training-c/"
    y_train_c_path = "Dataset/training-c.csv"
    x_train_d_path = "Dataset/training-d/"
    y_train_d_path = "Dataset/training-d.csv"
    x_train_e_path = "Dataset/training-e/"
    y_train_e_path = "Dataset/training-e.csv"

    ######################################### Build the model #########################################
    # LeNet Architecture
    model_lenet = CNN()
    model_lenet.add(Conv2D(6, 5, 1, 2))
    model_lenet.add(ReLU())
    model_lenet.add(MaxPool(2, 2))
    model_lenet.add(Conv2D(16, 5, 1, 0))
    model_lenet.add(ReLU())
    model_lenet.add(MaxPool(2, 2))
    model_lenet.add(Flatten())
    model_lenet.add(FC(120))
    model_lenet.add(FC(84))
    model_lenet.add(FC(10))
    model_lenet.add(Softmax())
    model_lenet.compile(log_loss, categorical_crossentropy_prime)

    ######################################### Load the data #########################################
    X_train, Y_train = load_train_dataset(x_train_a_path, y_train_a_path, samples_to_load=1000)
    print("X sample set shape: ", X_train.shape)
    print("Y sample set shape: ", Y_train.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)
    print("X train shape: ", X_train.shape)
    print("Y train shape: ", Y_train.shape)
    print("X validation shape: ", X_val.shape)
    print("Y validation shape: ", Y_val.shape)

    ######################################### Train the model #########################################
    batch_size = 32
    epochs = 10
    learning_rate = 0.01

    num_batches = math.ceil(X_train.shape[0] / batch_size)

    losses = []
    accuracies = []
    f1_scores = []
    for epoch in tqdm(range(epochs)):
        for batch in range(num_batches):
            print("Epoch: ", epoch, " Batch: ", batch)
            X_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
            Y_batch = Y_train[batch * batch_size: (batch + 1) * batch_size]

            loss = model_lenet.fit(X_batch, Y_batch, learning_rate)
            losses.append(loss)
            print("Loss: ", loss)

            accuracy, score, confusion = model_lenet.evaluate(X_val, Y_val)
            accuracies.append(accuracy)
            f1_scores.append(score)
            print("Accuracy: {}" .format(accuracy * 100))
            print("F1 Score: {}" .format(score))
            print("Confusion Matrix: {}" .format(confusion))

    x_epoch = np.arange(len(losses))

    plot_dir = "./"
    plt.figure(figsize=(10, 5))
    plt.plot(x_epoch, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.savefig(plot_dir + str(epochs) + "_" + str(learning_rate) + "_" + str(int(accuracy * 100))+"_loss.jpg", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x_epoch, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.savefig(plot_dir + str(epochs) + "_" + str(learning_rate) + "_" + str(int(accuracy * 100))+"_accuracy.jpg", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x_epoch, f1_scores)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Epoch")
    plt.savefig(plot_dir + str(epochs) + "_" + str(learning_rate) + "_" + str(int(accuracy * 100))+"_f1_score.jpg", bbox_inches='tight')
    plt.show()

    # # Save the model
    model_lenet.clear_cache()
    pickle.dump(model_lenet, open("1705108_model" + str(int(accuracy * 100))+ ".pkl", "wb"))

if __name__ == "__main__":
    main()