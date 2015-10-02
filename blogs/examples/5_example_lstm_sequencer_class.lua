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

   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end
