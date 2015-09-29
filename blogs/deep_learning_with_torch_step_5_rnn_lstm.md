The Long way of Deep Learning with Torch: part 5
============
**Abstract:** In this post we analyze how to use [rnn]() library to build a RNN and a LSTM based Neural Network.

[rnn](https://github.com/Element-Research/rnn#rnn.LSTM) is a torch library that can be used to implement RNNs, LSTMs, BRNNs, BLSTMs.



## AbstractRecurrent
All the classes inherited from the class `AbstractRecurrent`

```lua
rnn = nn.AbstractRecurrent(rho)
```
This class:

- takes as parameter `rho` which is the maximum number of steps to backpropagate through time (BPTT). The default value for `rho` is 9999 and means that the effect of the network is backpropagated through the entire sequence whatever its length.

	>Lower values of rho are useful when you have long sequences and you want to propagate only at least rho steps.
	>This is not valid for LSTM where the model learn when remember/forget.

- A step value is incremented each time a forward is called. When the current step number is equal to `rho` the forget function should be called.

- accepts as input a table of examples. **Thus, it can be trained using mini-batch transparently**

## RNN

`nn.Recurrent(start, input, feedback, [transfer, rho, merge])` takes 5 arguments:

- **start**: the size of the output, or a Module that will be inserted between the input and the transfer.
- **input**: a module that processes the input tensor.
- **feedback**: a module that feedbacks the previous output tensor
- **transfer**: a non-linear Module used to process the element-wise sum of the input and feedback module outputs
- **rho**: is the maximum amount of backprogragation.
- **merge**: a table Module that merges the outputs of the input and feedback

the transfer function and the merge function can be passed optionally.

A **forward** keeps a log of intermediate steps and increase the step 1 by 1. Back propagation through time is performed when `updateParameters` or `backwardThroughTime` method is called.  Note that the longer the sequence the more memory will be required to store all the output and gradInput states (one for each time step).

>**Suggestion**: To use this module with batches, we suggest using different sequences of the same size within a batch and calling updateParameters every rho steps and forget at the end of the Sequence.


**Example**
In the following an example an RNN based model and a function to update the gradient. 

```lua
model = nn.Sequential()
model:add(nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
))
model:add(nn.Linear(hiddenSize, nIndex))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
```

```lua
function gradientUpgrade(model, x, y, criterion, learningRate, i)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
	local gradOutputs = criterion:backward(prediction, y)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   model:backward(x, gradOutputs)

   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end

   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      model:backwardThroughTime()
      model:updateParameters(learningRate)
      model:zeroGradParameters()
   end
end
```

Finally, checks the complete [example](./examples/rnn_base.lua).

## Decorate with Sequencer

Any `AbstractRecurrent` instance can be decorated with a **Sequencer** such that an entire sequence of size `rho` can be presented with a single forward/backward call. The main differences with respect to the previous example are in model definition and inputs.

- Each layer in the model is annotated with sequencer:

	```lua
	-- Model
	model = nn.Sequential()
	model:add(nn.Sequencer(nn.Recurrent(
	   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
	   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
	   rho
	)))
	model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
	model:add(nn.Sequencer(nn.LogSoftMax()))
	
	criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
	```
- The inputs are now presented as a table with size `batchSize` of tensor of size `rho`.

	```lua	
	for i = 1, 10e4 do
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
	      -- a batch of targets
	      table.insert(targets, dataset:index(1, offsets))
	   end
	
	   i = gradientUpgrade(model, inputs, targets, criterion, lr, i)
	end		
	``` 

Checks [rrn with sequencer](./examples/rnn_sequencer.lua) for a complete example.

## LSTM and FastLSTM

The `nn.LSTM(inputSize, outputSize, [rho])` constructor takes 3 arguments:

- inputSize : a number specifying the size of the input;
- outputSize : a number specifying the size of the output;
- rho : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999.

![LSTM](https://github.com/Element-Research/rnn/raw/master/doc/image/LSTM.png)

```
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z(t)                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
Checks the [official doc](https://github.com/Element-Research/rnn#rnn.LSTM) for a complete explanation. In the following we us `FastLSTM` that performs the computation of input, forget and output gates together.

### Example
Let us build the same example as before with a LSTM.

```
require 'rnn'

batchSize = 10
rho = 5
hiddenSize = 64
nIndex = 10000


function gradientUpgrade(model, x, y, criterion, learningRate, i)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end
   i = i + 1
	local gradOutputs = criterion:backward(prediction, y)
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
   model:zeroGradParameters()
end


-- Model
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(nIndex, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


-- dummy dataset (task predict the next item)
dataset = torch.randperm(nIndex)
-- this dataset represent a random permutation of a sequence between 1 and nIndex

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
i = 1
for i = 1, 10000 do
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
      -- a batch of targets
      table.insert(targets, dataset:index(1, offsets))
   end

   i = gradientUpgrade(model, inputs, targets, criterion, lr, i)
end
```

The only difference with the RNN implementation is in the model.

```
-- Model
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(nIndex, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
```

It is composed by a LookupTable, a LSTM, a linear and a LogSoftMax to get the classification result. In the [examples folder](./examples) there is also a simple LSTM for regression and classification.
