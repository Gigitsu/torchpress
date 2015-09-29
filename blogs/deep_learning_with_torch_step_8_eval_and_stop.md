The Long way of Deep Learning with Torch: part 8
============
**Abstract:** In this post we analyze how to evaluate the model during the training and when to stop the training.

## Introduction
At this point we learn how to train a deep neural network on a given dataset. But we didn't talk abount learning. There is s subtle difference beetwen [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) and [learning](https://en.wikipedia.org/wiki/Learning), which subsume the keyword [overfitting](https://en.wikipedia.org/wiki/Overfitting).

When we train a neural network we are used to present the same examples several times. We need a method to evaluate how the network parameters are learning and when to stop the training.
To achieve these goals we will talk about:

- **training, validation and test set**: to split the initial dataset into 3 parts of size { 0.7, 0.1 , 0.2 }
- **evaluation method**: we need a method to compute the error of a given model with respect to a dataset.
- **early stop**: define a criteria to stop the learning after x mini-batches if the error on the validation-set is not decreasing
 

### Training, Validation and Test set
It can be achieved in two ways:

1. passing 3 separated datasets as input
2. defining a method that return 3 dataset of a given size

Since, Lua is not the best language to perform preprocessing operations I suggest to use the first option.


### Evaluation Method


