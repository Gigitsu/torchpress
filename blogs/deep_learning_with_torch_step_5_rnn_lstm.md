The Long path to Deep Learning using Torch: part 5
============
**Abstract:** In this post we analyze how to use [rnn]() library to build a RNN and a LSTM based Neural Network.

[rnn](https://github.com/Element-Research/rnn#rnn.LSTM) is a torch library that can be used to implement RNNs, LSTMs, BRNNs, BLSTMs.



## AbstractRecurrent
All the classes inherited from the class `AbstractRecurrent`

```
rnn = nn.AbstractRecurrent(rho)
```
This class takes as parameter `rho` which is the maximum number of steps to backpropagate through time (BPTT). The default value for `rho` is 9999 and means that the effect of the network is backpropagated through the entire sequence whatever its length.
>
>Lower values of rho are useful when you have long sequences and you want to propagate at least rho steps.

The step is incremented each time a forward is called. After the step number is compared to `rho` for forgetting.

## RNN

`nn.Recurrent(start, input, feedback, [transfer, rho, merge])` takes 5 arguments:

- start: the size of the output, or a Module that will be inserted between the input and the transfer.
- input: a module that processes the input tensor
- rho: is the maximum amount of backprogragation.

A **forward** keeps a log of intermediate steps and increase the step of 1. Back propagation through time is performed when `updateParameters` or `backwardThroughTime` method is called.  Note that the longer the sequence, the more memory will be required to store all the output and gradInput states (one for each time step).

>To use this module with batches, we suggest using different sequences of the same size within a batch and calling updateParameters every rho steps and forget at the end of the Sequence.

**Example**

```


```

## Decorate with Sequencer

It allows to present an entire sequence in a single call. Forward, backward and updateParameters are all that is required  ( `Sequencer` handles the `backwardThroughTime` internally ). Moreover it allows to pass to the model a batch of sequence to be trained in parallel.

### Sequence Effect

A sequencer is a kind of *Decorator* used to abstract away the complexity of `AbstractRecurrent` modules. 

- the Sequencer forwards an input sequence (a table) into an output sequence (a table of the same length). 
- It also takes care of calling forget, backwardThroughTime and other such AbstractRecurrent-specific methods.

For example the following two examples are equivalent:

```
input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
rnn:forward(input[1])
rnn:forward(input[2])
rnn:forward(input[3])
```

```
seq = nn.Sequencer(rnn)
seq:forward(input)
```

### Example RNN with Sequencer

```
require 'rnn'

batchSize = 10
rho = 5
hiddenSize = 10
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


--rnn layer
rnn = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
)

-- Model
model = nn.Sequential()
model:add(nn.Sequencer(rnn))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


-- dummy dataset (task predict the next item)
dataset = torch.randperm(nIndex)
-- this dataset represent a random permutation of a sequence between 1 and nIndex

-- define the batches in term of offset among the batches
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

If we print the x and y passed to the function `gradientUpgrade` we can see that they are a table of size 5 with a tensor of size 8 per row. Thus the RNN is trained using a batch of size 5 with 8 elements.

```
{
  1 : DoubleTensor - size: 8
  2 : DoubleTensor - size: 8
  3 : DoubleTensor - size: 8
  4 : DoubleTensor - size: 8
  5 : DoubleTensor - size: 8
}
{
  1 : DoubleTensor - size: 8
  2 : DoubleTensor - size: 8
  3 : DoubleTensor - size: 8
  4 : DoubleTensor - size: 8
  5 : DoubleTensor - size: 8
}
```

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
For formulas detail please check the [official doc](https://github.com/Element-Research/rnn#rnn.LSTM).

### Example
let us build the same example as before with a LSTM.

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
model:add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize, rho)))
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
model:add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
```

It is composed by a LookupTable, a LSTM, a linear and a LogSoftMax to get the classification result.