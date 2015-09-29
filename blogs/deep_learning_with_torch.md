Deep Learning with Torch: Aproaches
==========================

Now it is a month that I'm playing with Torch and Lua to define Deep Learning algorithms and I want to share my feelings and suggestions on it.

## What is Deep Learning

Broadly speaking, the goal of deep learning is to model complex, hierarchical features in data. "Deep learning" is not a particular type of algorithm, such as feedforward feedforward neural networks (FF nets for short) or SVMs, but rather a set of machine learning algorithms. In fact, any learning algorithm that learns a distributed representation of its input data could be considered a "deep learning algorithm". These algorithms can be used in both unsupervised and supervised learning.

**FF nets** are useful in deep learning problems because they consist of hidden units *h1…hn*, where each can learn an increasingly high-level representation of data. The input layer mapping from one layer to the next can vary from algorithm to algorithm. In [Deep belief networks](http://en.wikipedia.org/wiki/Deep_belief_network, the units are [Restricted Boltzmann machines](http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine). A very powerful deep learning algorithm is the [Convolutional Neural Network](http://en.wikipedia.org/wiki/Convolutional_neural_network). This learning algorithm takes advantage of pooling layers, which combine the outputs of neuron clusters, and shared weights. These algorithms often are comparatively computationally efficient, while still having a low error rate:

> As of 2011, the state of the art in deep learning feedforward networks alternates convolutional layers and max-pooling layers, topped by several pure classification layers. Training is usually done without any unsupervised pre-training. Since 2011, GPU-based implementations of this approach won many pattern recognition contests, including the IJCNN 2011 Traffic Sign Recognition Competition, the ISBI 2012 Segmentation of neuronal structures in EM stacks challenge, and others.

(Source: [Deep learning](http://en.wikipedia.org/wiki/Deep_learning#Deep_learning_in_artificial_neural_networks) article from Wikipedia).

If you're looking for a technical introduction to deep architectures, I recommend this review paper by Yoshua Bengio: [Page on iro.umontreal.ca](http://www.iro.umontreal.ca/~bengioy/papers/ftml.pdf).

Here is another great resource on deep learning, with links to a reading list, datasets, and software: [Deep Learning](http://deeplearning.net/).

### Introduction

As first step I started from [char-rnn project](). I didn't know lua nor torch and it was wery hard to understand what was in the code. I spent the first 2 weeks rewriting the code and doing experiments to get used to the language.
I was sure to be ready and I have implemented a complete revisitation of the Deep Neural Network. 
However, I found the LSTM model such as gibberish. This is because I decided to write a blog series titled "The Long way of Deep Learning with Torch". 
