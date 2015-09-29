The Long way of Deep Learning with Torch: part 3
============
**Abstract:** In this post we analyze how to use Criterions to build complex neural networks.

[Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md#criterions) are used to compute a gradient according to a given loss function given an input and a target. Criterions can be grouped into:

* Classification
* Regression
* Embedding criterions
* Miscelaneus criterions

Like the Module class, Criterions is an abstract class with the functions:

1. **forward(predicted, actualTarget)**: given a predicted value and a actual target computes the loss function associated to the criterions. 
2. **backward(input,target)**: given an input and a target compute the gradients of the loss function associated to the criterion.

These two function should be used to compute the loss and update the weights of the neural network layers.

## Criterions Examples

Let us give some examples to understand how criterions can be used.

### Classification: The negative log likelihood criterion

```
criterion = nn.ClassNLLCriterion([weights])
```
It is used to train a classificator on `n` classes. It takes optionally a 1 dimension tensor of `weights` which is useful if you have an unbalanced training set. 

The `forward` must have as input a *log-prograbilities * of each class. Log-probabilities can be obtained by appending a `nn.LogSoftMax` as last layer of your container.
The loss can be described as:

```
loss(x, class) = -weights[class] * x[class]
```

The following code fragment show how to use of the criterion to perform a gradient step. *This is the function that we should pass to an optimizer*

```lua
function gradientUpdate(model, x, y, learningRate)
   local criterion = nn.ClassNLLCriterion()
   local prediction = model:forward(x)
   local err = criterion:forward(prediction, y)
   model:zeroGradParameters()
   local gradient = criterion:backward(prediction, y)
   model:backward(x, gradient)
   model:updateParameters(learningRate)
end
```

### Regression: MSECriterion
```
criterion = nn.MSECriterion()
```
Creates a criterion that measures the mean squared error between n elements in the input x and output y:

```
loss(x, y) = 1/n \sum |x_i - y_i|^2 

```
If x and y are d-dimensional Tensors with a total of n elements, the sum operation still operates over all the elements, and divides by n. The two Tensors must have the same number of elements (but their sizes might be different).
 

### Other Criterions

  * Classification criterions:
    * [`BCECriterion`](#nn.BCECriterion): binary cross-entropy for [`Sigmoid`](transfer.md#nn.Sigmoid) (two-class version of [`ClassNLLCriterion`](#nn.ClassNLLCriterion));
    * [`ClassNLLCriterion`](#nn.ClassNLLCriterion): negative log-likelihood for [`LogSoftMax`](transfer.md#nn.LogSoftMax) (multi-class);
    * [`CrossEntropyCriterion`](#nn.CrossEntropyCriterion): combines [`LogSoftMax`](transfer.md#nn.LogSoftMax) and [`ClassNLLCriterion`](#nn.ClassNLLCriterion);
    * [`MarginCriterion`](#nn.MarginCriterion): two class margin-based loss;
    * [`MultiMarginCriterion`](#nn.MultiMarginCriterion): multi-class margin-based loss;
    * [`MultiLabelMarginCriterion`](#nn.MultiLabelMarginCriterion): multi-class multi-classification margin-based loss;
  * Regression criterions:
    * [`AbsCriterion`](#nn.AbsCriterion): measures the mean absolute value of the element-wise difference between input;
    * [`MSECriterion`](#nn.MSECriterion): mean square error (a classic);
    * [`DistKLDivCriterion`](#nn.DistKLDivCriterion): Kullback–Leibler divergence (for fitting continuous probability distributions);
  * Embedding criterions (measuring whether two inputs are similar or dissimilar):
    * [`HingeEmbeddingCriterion`](#nn.HingeEmbeddingCriterion): takes a distance as input;
    * [`L1HingeEmbeddingCriterion`](#nn.L1HingeEmbeddingCriterion): L1 distance between two inputs;
    * [`CosineEmbeddingCriterion`](#nn.CosineEmbeddingCriterion): cosine distance between two inputs;
  * Miscelaneus criterions:
    * [`MultiCriterion`](#nn.MultiCriterion) : a weighted sum of other criterions each applied to the same input and target;
    * [`ParallelCriterion`](#nn.ParallelCriterion) : a weighted sum of other criterions each applied to a different input and target;
    * [`MarginRankingCriterion`](#nn.MarginRankingCriterion): ranks two inputs;

## A Complete Example
In the following example we train a neural network for a classification task. The function `gradientUpgrade` performs one gradient step (forward, backward with update parameter.

```
require 'nn'

function gradientUpgrade(model, x, y, criterion, learningRate)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
	local gradOutputs = criterion:backward(prediction, y)
	model:zeroGradParameters()
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
end

model = nn.Sequential()
model:add(nn.Linear(5,1))

x1 = torch.rand(5)
y1 = torch.Tensor({1})
x2 = torch.rand(5)
y2 = torch.Tensor({-1})

criterion = nn.MarginCriterion(1)

for i = 1, 1000 do
	gradientUpgrade(model, x1, y1, criterion, 0.01)
	gradientUpgrade(model, x2, y2, criterion, 0.01)
end

-- with y1[1] we extract the first value in the tensor
print('prediction for x1 = ' .. model:forward(x1)[1] .. ' expected value ' .. y1[1])
print('prediction for x2 = ' .. model:forward(x2)[1] .. ' expected value ' .. y2[1])

print('loss after training for x1 = ' .. criterion:forward(model:forward(x1), y1))
print('loss after training for x2 = ' .. criterion:forward(model:forward(x2), y2))


```