require 'rnn'

batchSize = 50
rho = 5
hiddenSize = 64
nIndex = 10000

-- Model
model = nn.Sequential()
model:add(nn.LookupTable(nIndex, hiddenSize))
model:add(nn.FastLSTM(hiddenSize, hiddenSize, rho))
model:add(nn.Linear(hiddenSize, nIndex))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- dummy dataset (task is to predict next item, given previous)
dataset = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

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


lr = 0.01
for i = 1, 10e4 do
   local inputs = dataset:index(1, offsets)
   -- shift of one batch indexes
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   local targets = dataset:index(1, offsets)
   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end
