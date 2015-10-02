The Long way of Deep Learning with Torch: part 7
============
**Abstract:** In this post we analyze how to use **optim** to train a neural network.

## Overview

This package implements several optimization methods that can be used to train a neural network.

In the previous posts we showed how to train a neural network using a for and a learning function. The method `gradientUpgrade` peforms a learning step, which consists fo forward, backward and upgrade network weights.

```lua
function gradientUpgrade(model, x, y, criterion, learningRate, i)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end
	local gradOutputs = criterion:backward(prediction, y)
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
   model:zeroGradParameters()
end
```

However, the `Optim` package provides a complete list of optimization algorithms such as:

- [adadelta](https://github.com/torch/optim/blob/master/adadelta.lua)
- [adagrad](https://github.com/torch/optim/blob/master/adagrad.lua)
- [adam](https://github.com/torch/optim/blob/master/adam.lua)
- [asgd](https://github.com/torch/optim/blob/master/asgd.lua)
- [fista](https://github.com/torch/optim/blob/master/fista.lua)
- [lbfgs](https://github.com/torch/optim/blob/master/lbfgs.lua)
- [lswolfe](https://github.com/torch/optim/blob/master/lswolfe.lua)
- [rmsprop](https://github.com/torch/optim/blob/master/rmsprop.lua)
- [rprop](https://github.com/torch/optim/blob/master/rprop.lua)
- [sgd](https://github.com/torch/optim/blob/master/sgd.lua)

which can be used to train a nerual network.


Each optimization method is based on the same interface:

```lua
w_new,fs = optim.method(func,w,state)
```

where:

- **w_new** is the new parameter vector (after optimization),
- **fs** is a a table containing all the values of the objective, as evaluated during the optimization procedure:
	- **fs[1]** is the value before optimization, and
	- **fs[#fs]** is the most optimized one (the lowest).
- **func** is a closure function with the following interface: `f,df_dw = func(w)`, normally called `feval`
- **w** is the trainable/adjustable parameter vector,
- **state** a list of parameters algorithm dependent

Each method has a list of parameters that can be check on the source code.
Below there is a simple example of training with optim

```lua
algo_params = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i,sample in ipairs(training_samples) do
    local feval = function(w_nex)
       -- define eval function

       return loss_mini_batch,dl_d_mini_batch
    end
    optim.sgd(fevale,x,algo_params)
end
```

# Example LSTM with SGD

We take the example [lstm with sequencer](./examples/lstm_sequencer.lua) and replace the `iteration for` and the `gradientUpdate` with a `feval` function.

```lua
require 'rnn'
require 'optim'

batchSize = 50
rho = 5
hiddenSize = 64
nIndex = 10000

-- define the model
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(nIndex, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

```
Defines the model decorated with a `Sequencer`. Note that the criterion is decorated with `nn.SequenceCriterion`.

------------

```lua
-- create a Dummy Dataset, dummy dataset (task predict the next item)
dataset = torch.randperm(nIndex)

-- offset is a convenient pointer to iterate over the dataset
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)


-- method to compute a batch
function nextBatch()
	local inputs, targets = {}, {}
   for step = 1, rho do
      --get a batch of inputs
      table.insert(inputs, dataset:index(1, offsets))
      -- shift of one batch indexes
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      -- fill the batch of targets
      table.insert(targets, dataset:index(1, offsets))
   end
	return inputs, targets
end

```
Defines:

- a dummy dataset composed of a random permutation from 1 to nIndex,
- an offset table to store a list of pointers to scan the dataset, and
- a method to get the next batch.

------------

```lua
-- get weights and loss wrt weights from the model
x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
feval = function(x_new)
	-- copy the weight if are changed
	if x ~= x_new then
		x:copy(x_new)
	end
	-- select a training batch
	local inputs, targets = nextBatch()
	-- reset gradients (gradients are always accumulated, to accommodate
	-- batch methods)
	dl_dx:zero()

	-- evaluate the loss function and its derivative wrt x, given mini batch
	local prediction = model:forward(inputs)
	local loss_x = criterion:forward(prediction, targets)
	model:backward(inputs, criterion:backward(prediction, targets))

	return loss_x, dl_dx
end

```

Get the parameters from the built model and defines the feval function for the optimizer method

--------------

```lua
sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- cycle on data
for i = 1,1e4 do
	-- train a mini_batch of batchSize in parallel
	_, fs = optim.sgd(feval,x, sgd_params)

	if sgd_params.evalCounter % 100 == 0 then
		print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
		-- print(sgd_params)
	end
end

```
Defines the parameter for the method and the main for to perform mini-batches on the dataset. Each 100 mini-batches we prin the error.

Check the runnable examples for [classification](./examples/7_example_lstm_optim_class.lua) and [regression](./examples/7_example_lstm_optim_regr.lua).
