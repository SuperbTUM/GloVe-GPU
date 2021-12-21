import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
# assuming we have 4-gram in our model


def load_dataset():
    data_location = '../data.pk'
    data = pickle.load(open(data_location, 'rb'))
    return data


def log_cooccurence(word_data, V, symm=False):
    cooccurence_matrix = np.zeros((V, V))
    for words in word_data:
        cooccurence_matrix[words[0], words[1]] += 1
        cooccurence_matrix[words[1], words[2]] += 1
        cooccurence_matrix[words[2], words[3]] += 1
        if symm:
            cooccurence_matrix[words[1], words[0]] += 1
            cooccurence_matrix[words[2], words[1]] += 1
            cooccurence_matrix[words[3], words[2]] += 1
    smooth = 0.5
    cooccurence_matrix += smooth
    cooccurence_matrix = np.log(cooccurence_matrix)
    return cooccurence_matrix


def init(V, d):
    base = 0.1
    W = base * np.random.normal(size=(V, d))
    W_tilde = base * np.random.normal(size=(V, d))
    b = base * np.random.normal(size=(V, 1))
    b_tilde = base * np.random.normal(size=(V, 1))
    return W, W_tilde, b, b_tilde


def grad(W, W_tilde, b, b_tilde, co_occurence):
    V = co_occurence.shape[0]
    the_loss = W @ W_tilde.T + b @ np.ones([1, V]) + np.ones([V, 1]) @ b_tilde.T - co_occurence
    grad_W = 2 * (the_loss @ W_tilde)
    grad_W_tilde = 2 * (W.T @ the_loss).T
    grad_b = 2 * (np.ones([1, V]) @ the_loss).T
    grad_b_tilde = 2 * (np.ones([1, V]) @ the_loss).T
    return grad_W, grad_W_tilde, grad_b, grad_b_tilde


def loss(W, W_tilde, b, b_tilde, co_occurence):
    V = co_occurence.shape[0]
    return np.sum((W @ W_tilde.T + b @ np.ones([1, V]) + np.ones([V, 1]) @ b_tilde.T - co_occurence)**2)


def lr_scheduler(lr, epoch, drop=0.5, epoch_drop=5):
    return lr * drop ** (epoch // epoch_drop)


def train(W, W_tilde, b, b_tilde, V, shuffle=True):
    co_occurence_valid = log_cooccurence(data['valid_inputs'], V, symm=False)
    learning_rate = 0.05 / V
    momentum = 0.9
    epochs = 25
    batch_size = 74500
    train_losses, valid_losses = [], []
    step_w = step_w_tilde = step_b = step_b_tilde = 0
    if shuffle:
        data_inputs = data['train_inputs']
        num_batches = data_inputs.shape[0] // batch_size
        for epoch in range(epochs):
            idxs = np.random.permutation(data_inputs.shape[0])
            data_inputs_random = data_inputs[idxs, :]
            co_occurence_train = log_cooccurence(data_inputs_random, V, symm=False)
            learning_rate = lr_scheduler(learning_rate, epoch)
            for m in range(num_batches):
                data_inputs_batch = data_inputs_random[m * batch_size:(m + 1) * batch_size, :]
                co_occurence_train_batch = log_cooccurence(data_inputs_batch, V)
                grad_W, grad_W_tilde, grad_b, grad_b_tilde = grad(W, W_tilde, b, b_tilde, co_occurence_train_batch)
                step_w = step_w * momentum + learning_rate * grad_W
                W -= step_w
                step_w_tilde = step_w_tilde * momentum + learning_rate * grad_W_tilde
                W_tilde -= step_w_tilde
                step_b = step_b * momentum + learning_rate * grad_b
                b -= step_b
                step_b_tilde = step_b_tilde * momentum + learning_rate * grad_b_tilde
                b_tilde -= step_b_tilde

            train_loss = loss(W, W_tilde, b, b_tilde, co_occurence_train)
            train_losses.append(train_loss)

            valid_loss = loss(W, W_tilde, b, b_tilde, co_occurence_valid)
            valid_losses.append(valid_loss)
    else:
        raise Exception('Training data has to be shuffled!')
    final_W = W
    epoch_list = [i for i in range(1, epochs+1)]
    plt.figure()
    plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.plot(epoch_list, train_losses, label='training loss', linewidth=2)
    plt.plot(epoch_list, valid_losses, label='validation loss', linewidth=2)
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")
    plt.show()

    return final_W


def plot_tsne(final_W, word_data):
    mapped = TSNE(n_components=2).fit_transform(final_W)
    vocabs = word_data["vocab"]
    plt.figure(figsize=(12, 12))
    for i, w in enumerate(vocabs):
        plt.text(mapped[i, 0], mapped[i, 1], w)
    plt.xlim(mapped[:, 0].min(), mapped[:, 0].max())
    plt.ylim(mapped[:, 1].min(), mapped[:, 1].max())
    plt.savefig("glove.png")
    plt.show()


def main(data):
    V = len(data['vocab'])
    d = 10
    W, W_tilde, b, b_tilde = init(V, d)
    return train(W, W_tilde, b, b_tilde, V)


if __name__ == '__main__':
    data = load_dataset()
    start = time.time()
    final_W = main(data)
    end = time.time()
    print("CPU execution time is {:.2f} seconds.".format(end-start))
    plot_tsne(final_W, data)

