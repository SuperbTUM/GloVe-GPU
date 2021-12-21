import numpy as np
import pickle
from naive_glove import lr_scheduler
import time
import matplotlib.pyplot as plt

TRAIN_CONFIG = {"batch_size": 100,
                "w_init": 0.01,
                "hidden_dim": 128,
                "embedding_dim": 16,
                "learning_rate": 0.1,
                "momentum": 0.9}


def logistic(y):
    try:
        return 1. / (1. + np.exp(-y))
    except RuntimeWarning:
        raise Exception("Overflow Encountered. Check your model!")


class Model(object):
    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim, w_init, context_length):
        # definition of hyper-parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.w_init = w_init
        self.context_length = context_length
        # definition of layers
        self.embedding_layer = np.zeros((self.batch_size, self.context_length * self.embedding_dim))
        self.hidden_layer_activated = np.zeros((self.batch_size, self.hidden_dim))

        # definition of trainable parameters
        self.word_embedding_weights = np.random.normal(scale=self.w_init,
                                                       size=(self.vocab_size, self.embedding_dim))

        self.emb_to_hid_weights = np.random.normal(scale=self.w_init,
                                                   size=(self.hidden_dim, self.context_length * self.embedding_dim))
        self.hid_bias = np.zeros((self.hidden_dim,))

        self.hid_to_out_weights = np.random.normal(scale=self.w_init,
                                                   size=(self.context_length * self.vocab_size, self.hidden_dim))
        self.out_bias = np.zeros((self.context_length * self.vocab_size,))

    def _softmax(self, y):
        y = np.exp(y)
        y_shape = y.shape
        y = y.reshape((-1, self.context_length, self.vocab_size))
        y /= y.sum(axis=-1, keepdims=True)
        y = y.reshape(y_shape)
        return y

    def forward(self, batch_data):

        for i in range(self.context_length):
            self.embedding_layer[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
                self.word_embedding_weights[batch_data[:, i], :]
        hidden_layer = np.dot(self.embedding_layer, self.emb_to_hid_weights.T) + self.hid_bias  # (B, Nd) @ (Nd, H) -> (B, H)
        self.hidden_layer_activated = logistic(hidden_layer)
        output_layer = np.dot(self.hidden_layer_activated, self.hid_to_out_weights.T) + self.out_bias  # (B, H) @ (H, NV) -> (B, NV)
        output_layer -= output_layer.max(1).reshape((-1, 1))
        output_layer_activated = self._softmax(output_layer)
        return output_layer_activated

    def indicator_matrix(self, batch_data, mask_zero=True):
        """
        Generate a (B, NV)-shape masked input batch data
        :param mask_zero:
        :param batch_data:
        :return:
        """
        batch_size, context_length = batch_data.shape
        target_batch = np.zeros((batch_size, context_length * self.vocab_size))
        targets_offset = np.repeat((np.arange(context_length) * self.vocab_size)[np.newaxis, :],
                                   batch_size, axis=0)
        batch_data += targets_offset
        for c in range(context_length):
            target_batch[np.arange(batch_size), batch_data[:, c]] = 1.
            if mask_zero:
                target_batch[np.arange(batch_size), targets_offset[:, c]] = 0.
        return target_batch

    @staticmethod
    def compute_loss(target_batch, output_activated):
        cross_entropy = -np.sum(target_batch * np.log(output_activated + 1e-5))
        return cross_entropy

    def compute_loss_derivative(self, target_batch, output_activated, mask):
        deriv = output_activated - target_batch
        deriv *= np.repeat(mask, self.vocab_size, axis=1)
        return deriv

    def backward(self, input_batch, loss_derivative):
        batch_size, NV = loss_derivative.shape
        hidden_deriv = np.dot(loss_derivative, self.hid_to_out_weights) * self.hidden_layer_activated * (1. - self.hidden_layer_activated)
        hid_to_output_weights_grad = loss_derivative.T @ self.hidden_layer_activated
        output_bias_grad = loss_derivative.T @ np.ones([batch_size, ])
        embed_to_hid_weights_grad = hidden_deriv.T @ self.embedding_layer
        hid_bias_grad = hidden_deriv.T @ np.ones([batch_size, ])
        embed_deriv = np.dot(hidden_deriv, self.emb_to_hid_weights)
        word_embedding_weights_grad = np.zeros((self.vocab_size, self.embedding_dim))
        for w in range(self.context_length):
            word_embedding_weights_grad += np.dot(
                self.indicator_matrix(input_batch[:, w:w + 1], mask_zero=False).T,
                embed_deriv[:, w * self.embedding_dim:(w + 1) * self.embedding_dim])

        return word_embedding_weights_grad, embed_to_hid_weights_grad, hid_bias_grad, hid_to_output_weights_grad, output_bias_grad

    def sample_input_mask(self):
        """Samples a binary mask for the inputs of size batch_size x context_len
        For each row, at most one element will be 1.
        """
        mask_idx = np.random.randint(self.context_length, size=(self.batch_size,))
        mask = np.zeros((self.batch_size, self.context_length),
                        dtype=np.int32)  # Convert to one hot B x N, B batch size, N context len
        mask[np.arange(self.batch_size), mask_idx] = 1
        return mask

    def evaluate(self, inputs):
        pass


def train():

    np.random.seed(1)

    data = pickle.load(open("data.pk", "rb"))
    data_inputs = data["train_inputs"]
    model = Model(TRAIN_CONFIG["batch_size"], len(data["vocab"]), TRAIN_CONFIG['embedding_dim'],
                  TRAIN_CONFIG['hidden_dim'], TRAIN_CONFIG["w_init"], data_inputs.shape[1])
    epochs = 10
    learning_rate = TRAIN_CONFIG["learning_rate"]
    batch_size = TRAIN_CONFIG["batch_size"]
    momentum = TRAIN_CONFIG["momentum"]
    num_batches = data_inputs.shape[0] // batch_size
    step_embedding_w = step_emb_to_hid = step_hid_bias = step_hid_to_output = step_out_bias = 0

    train_losses, cur_batches = [], []

    for epoch in range(epochs):
        idxs = np.random.permutation(data_inputs.shape[0])
        data_inputs_random = data_inputs[idxs, :]
        # data_inputs_random = data_inputs[:]
        learning_rate = lr_scheduler(learning_rate, epoch)
        train_loss = 0.
        for m in range(num_batches):
            # Data Preparation
            data_inputs_batch = data_inputs_random[m * batch_size: (m + 1) * batch_size, :]
            mask = model.sample_input_mask()
            input_batch_masked = data_inputs_batch * (1 - mask)  # We only zero out one word per row
            target_batch_masked = data_inputs_batch * mask
            target_batch = model.indicator_matrix(target_batch_masked)
            # forward
            output_activated = model.forward(input_batch_masked)
            # calculate loss derivative
            deriv = model.compute_loss_derivative(target_batch, output_activated, mask)
            deriv /= batch_size
            # calculate cross entropy loss
            train_loss += model.compute_loss(target_batch, output_activated) / batch_size
            # calculate parameter gradients
            word_embedding_weights_grad, embed_to_hid_weights_grad, hid_bias_grad, hid_to_output_weights_grad, output_bias_grad = model.backward(
                data_inputs_batch, deriv)
            # update parameters
            step_embedding_w = momentum * step_embedding_w + learning_rate * word_embedding_weights_grad
            model.word_embedding_weights -= step_embedding_w

            step_emb_to_hid = momentum * step_emb_to_hid + learning_rate * embed_to_hid_weights_grad
            model.emb_to_hid_weights -= step_emb_to_hid

            step_hid_bias = momentum * step_hid_bias + learning_rate * hid_bias_grad
            model.hid_bias -= step_hid_bias

            step_hid_to_output = momentum * step_hid_to_output + learning_rate * hid_to_output_weights_grad
            model.hid_to_out_weights -= step_hid_to_output

            step_out_bias = momentum * step_out_bias + learning_rate * output_bias_grad
            model.out_bias -= step_out_bias

        cur_batch = num_batches * (epoch+1)
        cur_loss = train_loss / num_batches
        cur_batches.append(cur_batch)
        train_losses.append(cur_loss)
        print("Iterations: {}, CE loss: {:.3f}".format(cur_batch, cur_loss))
    plt.figure()
    plt.plot(cur_batches, train_losses, linewidth=2)
    plt.xlabel("num of batches")
    plt.ylabel("train loss")
    plt.grid()
    plt.savefig("loss_nn.png")
    plt.show()


if __name__ == "__main__":
    start = time.time()
    train()
    end = time.time()
    print("CPU execution time is {:.2f} seconds.".format(end - start))
