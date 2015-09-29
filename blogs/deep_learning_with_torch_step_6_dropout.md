The Long way of Deep Learning with Torch: part 6
============
**Abstract:** In this post we analyze how to implement **Dropout**.

## Dropout
Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at
test time. Dropout is a technique for addressing this problem.

The key idea is to randomly drop units (along with their connections) from the neural network during training. It does so by “dropping out” some unit activations in a given layer, that is setting them to zero. This prevents units from co-adapting too much. 

- **During training**, dropout samples from an exponential number of different “thinned” networks. 
- **At test time**, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. 

This significantly reduces overfitting and gives major improvements over other regularization methods.

The simple example below shows how each input element has a probability of p of being dropped.

```lua
require 'nn'

module = nn.Dropout()

> x=torch.Tensor{{1,2,3,4},{5,6,7,8}}

> module:forward(x)
  2   0   0   8
 10   0  14   0
[torch.DoubleTensor of dimension 2x4]

> module:forward(x)
  0   0   6   0
 10   0   0   0
[torch.DoubleTensor of dimension 2x4]

```

Furthermore, the ouputs are scaled by a factor of 1/(1-p) during training.

### Using Dropout in Model Building

It can be used as layer between each layer of a model.

```lua

model = nn.Sequential()
model:add(nn.Linear(10, 25))
model:add(nn.Dropout())
model:add(nn.Tanh()) 
model:add(nn.Linear(25, 1))
```

Here we added a Dropout between the two linear models.