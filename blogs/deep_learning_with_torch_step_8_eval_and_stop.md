The Long way of Deep Learning with Torch: part 8
============
**Abstract:** In this post we analyze how to evaluate the model during the training and when to stop the training.

## Introduction
At this point we learn how to train a deep neural network on a given dataset. But we didn't talk abount learning. There is s subtle difference beetwen [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) and [learning](https://en.wikipedia.org/wiki/Learning), which subsume the keyword [overfitting](https://en.wikipedia.org/wiki/Overfitting).

When we train a neural network we are used to present the same examples several times. We need a method to evaluate how the network parameters are learning and when to stop the training.
To achieve these goals we talk about:

- **training, validation and test set**: to split the initial dataset into 3 parts of size { 0.7, 0.1 , 0.2 }
- **evaluation method**: we need a method to compute the error of a given model with respect to a dataset different from the training set.
- **early stop**: define a criteria that stops the training if the error on the training set is decreasing while it is increasing on the validation set.


### Training, Validation and Test set
It can be achieved in two ways:

1. passing 3 separated datasets as input
2. defining a method that return 3 dataset of a given size

Since, Lua is not the best language to perform preprocessing operations I suggest you, based on my experience, to provide 3 datasets.

### Early stop

In the example [lstm with validation](./examples/8_example_lstm_validation.lua) is provided a method to check the error on a validation set. This method can be used to implement an early stop criteria.
There are different way to implement early stop as described in the paper [Early Stopping - but when?](http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf).
