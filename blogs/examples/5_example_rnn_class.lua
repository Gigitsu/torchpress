require 'rnn'

batchSize = 10
rho = 5
-- used to call the BPTT
updateInterval = 4
hiddenSize = 32
nIndex = 10000

-- Model
model = nn.Sequential()
model:add(nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
))
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
