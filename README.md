# Acceleration of GloVe Representation

## Contents

1. [Proposal](#proposal)
2. [Results](#results)
3. [Dataset & Checkpoint](#dataset--checkpoint)
3. [Environment](#environment)
3. [Installation](#Installation)
4. [Quick Start](#quick-start)
5. [Contact us](#contact-us)

## Proposal

There are many times we are more willing to leverage deep learning frameworks, for instance, PyTorch or Tensorflow, to train neural networks. Thanks to sophisticated design, both PyTorch and Tensorflow have integrated cuda and cudnn in its framework. With a simple switch, the network can be trained or tested via GPU. However, autograd is not always fast and reliable enough when it comes to complicated condition branches or iteration branches in network design.

The goal of the project is to re-implement GloVe representation with heterogeneous computing. The intuition behind this is that we find there are many matrix-level calculations, like multiplication, addition, reduction, and division. We also find there is something wrong with PyTorch since it's unable to do auto backward propagation accurately. The wisest thing to do is to accelerate numpy implementation of GloVe. To check the correctness of our project, we have the dataset comes from Department of Computer Science, University of Toronto. The amount of data is approx. 375,000 with 251 unique words. The naive implementation includes co-occurrence, embedding. The training process leverages data shuffle, batch gradient descent with momentum, and step learning rate scheduler. Final clustering is visualized with T-SNE package. 

In the first step, with PyCUDA, the project is expected to achieve parallelization in co-occurrence matrix, gradient update and loss calculation in one single epoch. The baseline (CPU numpy implementation) will take more than 56 seconds with 25 epochs to converge.

In the second step, with PyCUDA, the project is expected to achieve more complicated network parallelization. For n-gram dataset, we intend to have n outputs at a time. Assuming there is a three-layer network with embedding, hidden and output layer. The shape of embedding layer is (vocab_size, context_length * embedding_dim), the shape of hidden layer is (hidden_dim, ) and the shape of output_layer is (context_length * vocab_size, ). We apply sigmoid activation between hidden layer and output layer and SoftMax activation after output layer. The loss function is replaced by cross entropy loss. In inference mode, it takes 38 seconds.

Overall, we have our baseline as follows:

| Methodology     | naive version with numpy (Intel i7, 4 core) | neural network version with numpy |
| --------------- | ------------------------------------------- | --------------------------------- |
| Time in seconds | ≥ 56 seconds in 25 epochs                   | ≥ 38 seconds in a single epoch    |

In the second step, in addition to what we should have done in the first step, we need to make full use of shared memory to do reduction since there is a SoftMax operation. The core implementation will focus on parallelization of forward propagation of the network.

Overall, the project will concentrate on matrix-to-matrix calculation, matrix-to-vector calculation, reduction, and etc. The project will also try to figure out how to minimize running time by reducing memory transfer frequency.

## Results

We have achieved acceleration of naive GloVe representation. With cuBLAS and streams, the training time is **0.69** seconds for 25 epochs, compared with 56 seconds in numpy implementation with same number of epochs.

We also have achieved acceleration of neural network version of GloVe. The inference time is **8.92** seconds on average, compared with 38 seconds in numpy implementation.

## Dataset & Checkpoint

- Dataset can be accessed [here](https://drive.google.com/file/d/1B8Gr9G66ZRj6lvpVoVMWTyxDD52Awv1g/view?usp=sharing)


- Checkpoint can be accessed [here](https://drive.google.com/file/d/15Am6cbYhNBepm84h4MQtiXv8gO-N4A5A/view?usp=sharing)

## Environment

- Operating System: Linux 5.4
- CUDA: 11.3  
- GPU: Nvidia Tesla P100

## Installation

This project is based on numpy, pycuda, scikit-cuda and pytorch. The requirements are specified in `requirements.txt`, and you can install them with the the following command:

```
pip install -r requirements.txt
```

## Quick Start

Download the dataset and checkpoint and place them according to the path of `load_dataset` function in each Python script. The implementation with numpy, PyCUDA and PyTorch reside in 3 separated folders:

#### Numpy folder

This is the baseline folder where you can take it as reference. Try to run the code and record the time as well as the loss. Please note that the numpy version uses `numpy.float64` as core data type. Also, there exists randomness and precision loss that may cause slight difference of the loss. In the script of `glove_nn.py`, you can run the original code to see the correctness of the neural network design. To better compare with the GPU implementation, you can delete all the backward procedure and transform the code from training mode to inference mode.

- `naive_glove.py`: naive GloVe 
  Expected output: CPU execution time, plot of loss (`loss.png` under the directory), tsne plot of words (`glove.png` under the directory)
- `glove_nn.py`: neural network version of GloVe 
  Expected output: CPU execution time, plot of loss (`loss_nn.png` under the directory)

#### PyCUDA folder  

This is the core folder that we have been working on. Try to run the code and compare with what have been recorded in the numpy folder.

- `naive_glove_pycuda.py`: naive GloVe
  Expected output: GPU execution time
- `glove_nn_pycuda.py`: neural network version of GloVe
  Expected output: GPU execution time

#### PyTorch folder

This is the PyTorch implementation. For some reason, it couldn't convey correct backward propagation to us. We will take a look at this later.

- `glove_pytorch.py`: neural network version of Glove (CPU version)
  Expected output: PyTorch CPU implementation time
- `baseline_pytorch_cuda.ipynb`: neural network version of Glove (GPU version)
  Expected output: PyTorch GPU implementation time

## Contact us

Mingzhe: mh4116@columbia.edu

Wenpu: ww2569@columbia.edu
