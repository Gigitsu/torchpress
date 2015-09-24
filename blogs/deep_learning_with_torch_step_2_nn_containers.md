The Long path to Deep Learning using Torch: part 2
============
**Abstract:** In this post we analyze how to build complex neural networks using the container classes.

[Containers](https://github.com/torch/nn/blob/master/doc/containers.md#nn.Containers) such as [Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) is an abstract class that defines the main functions that must be inherited from concrete classes.

The main functions of Container are:

1. add(module): add a Module to the given container
2. get(index): get the module at index
3. size(): the size of the container

## Implemented Containers

### Sequential

Sequential provides a means to plug layers together in a feed-forward fully connected manner.

```lua

model = nn.Sequential()
model:add( nn.Linear(10, 25) ) -- 10 input, 25 hidden units
model:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
model:add( nn.Linear(25, 1) ) -- 1 output

print(model:forward(torch.randn(10)))

```

which gives as output

```lua
-0.1815
[torch.Tensor of dimension 1]
```

Moreover this container offers method to insert a module at index and remove a module at index.

### Parallel

Create a container that allows to train in parallel different layers. For example we can define a model composed of two parallel layers with the same input size. Their output is concatenated together.

```lua
model = nn.Parallel(2,1)
model:add(nn.Linear(10,3))
model:add(nn.Linear(10,2))
print(model:forward(torch.randn(10,2)))
```

gives as output a Tensor 5x1

### Concat

Concat concatenates the output of one layer of "parallel" modules along the provided dimension dim: they take the same inputs, and their output is concatenated.

```
model=nn.Concat(1);
model:add(nn.Linear(5,3))
model:add(nn.Linear(5,7))
print(model:forward(torch.randn(5)))
```

## Conclusion

With Module, Layers and Containers we have the basis to build Deep Neural Networks.