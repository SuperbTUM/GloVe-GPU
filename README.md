# Proposal

There are many times we are more willing to leverage deep learning frameworks, for instance, PyTorch or Tensorflow, to train neural networks. Thanks to sophisticated design, both PyTorch and Tensorflow have integrated cuda and cudnn in its framework. With a simple switch, the network can be trained or tested via GPU. However, autograd is not always fast and reliable enough when it comes to complicated condition branches or iteration branches in network design.

The goal of the project is to re-implement GloVe representation with heterogeneous computing. The amount of data is approx. 320,000 with approx. 250 unique words. The naive implementation includes co-occurrence, embedding. The training process leverages data shuffle, batch gradient descent with momentum, and step learning rate scheduler. Final clustering is visualized with T-SNE package. 

In the first step, with either CUDA or OpenCL, the project is expected to achieve parallelization in co-occurrence matrix, gradient update and loss calculation in one single epoch. The baseline (CPU numpy implementation) will take more than 30 seconds to complete.

In the second step, with either CUDA or OpenCL, the project is expected to achieve more complicated network parallelization. For n-gram dataset, we intend to have n outputs at a time. Assuming there is a three-layer network with embedding, hidden and output layer. The shape of embedding layer is (vocab_size, context_length * embedding_dim), the shape of hidden layer is (hidden_dim, ) and the shape of output_layer is (context_length * vocab_size, ). We apply sigmoid activation between hidden layer and output layer and Softmax activation after output layer. The loss function is replaced by cross entropy loss. There are three baselines in this implementation. (epoch=10)

| Methodology     | numpy in CPU (Intel i7, 4 core) | PyTorch in CPU | PyTorch in GPU (Tesla P100) |
| --------------- | ------------------------------- | -------------- | --------------------------- |
| Time in seconds | ≥ 300                           | approx. 200    | approx. 160 (ave 15%) occupation|

In this case, in addition to what we should have done in the first step, we need to make full use of shared memory to do reduction since there is a Softmax operation. The core implementation will focus on parallelization of forward and back propagation of the network.

Overall, the project will concentrate on matrix-to-matrix calculation, matrix-to-vector calculation, reduction, and etc. The project will also try to figure out how to minimize running time by reducing memory transfer frequency.

