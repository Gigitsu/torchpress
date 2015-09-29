The Long way of Deep Learning with Torch: part 1
============
**Abstract:** In this post we analyze the first step to build a Deep Neural Network using [torch](http://torch.ch/). In particular, we focus on [torch/nn module](torch/nn](https://github.com/torch/nn).



Module in [torch/nn](https://github.com/torch/nn) is an abstract class that includes the fundamental method necessary to define a Layer of a neural network.

Module has two state variable as describe in its `_init()` function.

```lua
function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end
```

An the methods to:

1. forward(input): takes an input object, and computes the corresponding output of the module.
2. backward(input, gradOutput): performs a backpropagation step through the module with respect to the given input. 
3. All the other function are defined in the [documentation](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module)

## Module Implementations

The abstract class Module is extended in several [Layers](https://github.com/torch/nn/blob/master/doc/simple.md#nn.simplelayers.dok). To describe the main functions provided by `Module` let us consider some Layers Implementations.

### Linear

Applies a linear transformation to the incoming data, i.e. `y = Ax + b`.

```lua
-- define a module
module = nn.Linear(10,5)

-- create a model
model = nn.Sequential()
model:add(module)

-- weights and biases 
print(module.weight)
print(module.bias)

-- the gradient
print(module.gradWeight) 
print(module.gradBias)
```

- **A forward step is computed by:**

	```lua
	x = torch.Tensor(10)
	y = model:forward(x) -- or module:forward(x), module(x)
	```

- **A backward step is done by:**

	```lua
	gradInput = model:backward(x, y)
	```
	This function simply performs this task using two function calls:
	- A function call to updateGradInput(input, gradOutput).
	- A function call to accGradParameters(input,gradOutput,scale).


### Identity
Creates a module that returns whatever is input to it as output.

```lua
module = nn.Identity()
model = nn.Sequential()
model:add(module)

x = torch.range(1,5)

model:forward(x) -- print x
```

this layer can be used to model the input layer of a neural network

Moreover, it offers layers to perform:

- **Add:** Applies a bias term to the incoming data, i.e. `yi = x_i + b_i`, or if `scalar = true` then uses a single bias term, `yi = x_i + b`.
- **Mul:** Applies a single scaling factor to the incoming data, i.e. `y = w x`, where `w` is a scalar.
- **CMul:** Applies a component-wise multiplication to the incoming data, i.e. `y_i = w_i * x_i`. For example, `nn.CMul(3,4,5)` is equivalent to `nn.CMul(torch.LongStorage{3,4,5})`.
- **Narrow:**Narrow is application of [narrow](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-narrowdim-index-size) operation in a module.
- **Reshape:**Reshapes an `nxpxqx..` Tensor into a `dimension1xdimension2x...` Tensor, taking the elements column-wise.

The other operations are described in the [documentation](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Max)

In the next series we will describe how to use Layers combined with [Containers](https://github.com/torch/nn/blob/master/doc/containers.md#nn.Containers)

