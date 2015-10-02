require 'rnn'

batchSize = 50
rho = 6
hiddenSize = 64
inputSize = 4
outputSize = 1
seriesSize = 10000

-- require('mobdebug').start()

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

model = nn.Sequential()
model:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))

criterion = nn.SequencerCriterion(nn.MSECriterion())


-- dummy dataset (task predict the next item)
dataset = torch.randn(seriesSize, inputSize)

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.001
for i = 1, 10e5 do
   local inputs, targets = {}, {}
   for step = 1, rho do
      --get a batch of inputs
      table.insert(inputs, dataset:index(1, offsets))
      -- shift of one batch indexes
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > seriesSize then
            offsets[j] = 1
         end
      end
      -- a batch of targets
      table.insert(targets, dataset[{{},1}]:index(1,offsets))
   end
   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end
