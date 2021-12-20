# Acceleration of GloVe Representation

## Contents

1. [Proposal](#proposal)
2. [Results](#results)
3. [Dataset & Checkpoint](#dataset--checkpoint)
3. [Installation](#Installation)
4. [Quick Start](#quick-start)
5. [Contact us](#contact-us)

## Proposal

There are many times we are more willing to leverage deep learning frameworks, for instance, PyTorch or Tensorflow, to train neural networks. Thanks to sophisticated design, both PyTorch and Tensorflow have integrated cuda and cudnn in its framework. With a simple switch, the network can be trained or tested via GPU. However, autograd is not always fast and reliable enough when it comes to complicated condition branches or iteration branches in network design.

The goal of the project is to re-implement GloVe representation with heterogeneous computing. Dataset comes from Department of Computer Science, University of Toronto. The amount of data is approx. 375,000 with 251 unique words. The naive implementation includes co-occurrence, embedding. The training process leverages data shuffle, batch gradient descent with momentum, and step learning rate scheduler. Final clustering is visualized with T-SNE package. 

In the first step, with either CUDA or OpenCL, the project is expected to achieve parallelization in co-occurrence matrix, gradient update and loss calculation in one single epoch. The baseline (CPU numpy implementation) will take more than 30 seconds to complete.

In the second step, with either CUDA or OpenCL, the project is expected to achieve more complicated network parallelization. For n-gram dataset, we intend to have n outputs at a time. Assuming there is a three-layer network with embedding, hidden and output layer. The shape of embedding layer is (vocab_size, context_length * embedding_dim), the shape of hidden layer is (hidden_dim, ) and the shape of output_layer is (context_length * vocab_size, ). We apply sigmoid activation between hidden layer and output layer and SoftMax activation after output layer. The loss function is replaced by cross entropy loss. There are three baselines in this implementation. (epoch=10)

| Methodology     | numpy in CPU (Intel i7, 4 core) | PyTorch in CPU | PyTorch in GPU (Tesla P100) |
| --------------- | ------------------------------- | -------------- | --------------------------- |
| Time in seconds | ≥ 300                           | approx. 200    | approx. 160 (ave 15%) occupation|

In this case, in addition to what we should have done in the first step, we need to make full use of shared memory to do reduction since there is a SoftMax operation. The core implementation will focus on parallelization of forward and back propagation of the network.

Overall, the project will concentrate on matrix-to-matrix calculation, matrix-to-vector calculation, reduction, and etc. The project will also try to figure out how to minimize running time by reducing memory transfer frequency.

## Results

We have achieved acceleration of naive GloVe representation. With cuBLAS and streams, the training time is 0.81 seconds for 25 epochs, compared with 56 seconds in numpy implementation with same number of epochs.

We also have achieved acceleration of neural network version of GloVe. The inference time is 15.62 seconds on average, compared with 38 seconds in numpy implementation.

## Dataset & Checkpoint

- Dataset can be accessed [here](https://drive.google.com/file/d/1B8Gr9G66ZRj6lvpVoVMWTyxDD52Awv1g/view?usp=sharing)


- Checkpoint can be accessed [here](https://drive.google.com/file/d/15Am6cbYhNBepm84h4MQtiXv8gO-N4A5A/view?usp=sharing)

## Installation

This project is based on numpy, pycuda, scikit-cuda and pytorch, please ensure you have installed them.

```
pip install numpy pycuda scikit-cuda
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Quick Start

Download the dataset from the link in Dataset & Checkpoint, numpy, pycuda and pytorch implementation reside in 3 separated folders, the content of folders is shown below:

#### numpy folder:  

- naive_glove.py: naive GloVe  

  Expected output: CPU execution time, plot of loss (loss.png under the directory), tsne plot of words (glove.png under the directory)

- glove_nn.py: neural network version of GloVe  

  Expected output: CPU execution time, plot of loss (loss_nn.png under the directory)

#### pycuda folder:  
- naive_glove_pycuda.py: naive GloVe  

  Expected output: GPU execution time

- glove_nn_pycuda.py: neural network version of GloVe  

  Expected output: GPU execution time

#### pytorch folder:
- glove_pytorch.py: neural network version of Glove (CPU version)  
  Expected output: PyTorch CPU implementation time
- baseline_pytorch_cuda.ipynb: neural network version of Glove (GPU version)  
  Expected output: PyTorch GPU implementation time

## Contact us

Mingzhe: mh4116@columbia.edu

Wenpu: ww2569@columbia.edu

