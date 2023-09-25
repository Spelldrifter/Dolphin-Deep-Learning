
# Dolphin - Deep Learning

This project is all about implementing deep neural networks on Intel multicore and MIC architecture which includes Intel Xeon Phi Coprocessors.

## Brief Summary

The core implementation revolves around the parallel Stacked Autoencoder and Restricted Boltzmann Machine (RBM) training algorithms on Intel Xeon & Xeon Phi platforms. In addition to this, we've developed a demo component that combines Stacked Autoencoders with a Softmax classifier neural network. For gradient computation, we use the Steepst Descent algorithm.

The repository uses a common code base for both the Intel Xeon (multi-core) and Intel Xeon Phi (many-core) platforms. To run the project on Xeon Phi, simply compile the program with the '`-mmic'` compiler option. Please be aware, performance optimizations specific to the Intel Xeon Phi platform may require modifications to the OpenMP parameters in the source code and `consts.h`.

For an in-depth description of our work and to examine our performance experiment results, please refer our published paper: [Training Large Scale Deep Neural Networks on the Intel Xeon Phi Many-core Coprocessor](http://pasa-bigdata.nju.edu.cn/people/ronggu/pub/DeepLearning_ParLearning.pdf).

## Compilation & Execution

The project requires Intel C/C++ Compiler and Intel MKL Library for compilation. The Makefile handles the compilation process. Subsequently, execute 'stackedAutoEncoder' for multi-core platform and 'stackedAutoEncoder.mic' for Xeon Phi.

## Customization

Backend configurations regarding the training & test dataset paths and DNN layers & nodes info can be customized in the `main.cpp`.

## Usage

To execute the compiled binary run `./stackedAutoEncoder` or `make run` under the same directory as of the data set.

For running the project natively on Xeon Phi platform, repeat the same process using the `stackedAutoEncoder.mic` file.

## Disclaimer

The source code in this repository is intended only for research and educational purposes. Any misuse of this code is not the responsibility of the author. The provided dataset is a subset of the [MNIST Database](http://yann.lecun.com/exdb/mnist/), please refer to the original source for usage guidelines.