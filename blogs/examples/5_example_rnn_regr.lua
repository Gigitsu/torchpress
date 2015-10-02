require 'rnn'

batchSize = 10
rho = 5
updateInterval = rho - 1
hiddenSize = 10
inputSize = 1
outputSize = 1

nIndex = 10000

model = nn.Sequential()
model:add(nn.Recurrent(
   hiddenSize, nn.Linear(inputSize,hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
))
model:add(nn.Linear(hiddenSize, outputSize))
criterion = nn.MSECriterion()

function gradientUpgrade(model, x, y, criterion, learningRate, i)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end
	local gradOutputs = criterion:backward(prediction, y)
	model:backward(x, gradOutputs)

	-- note that updateInterval < rho
	if i % updateInterval == 0 then
		-- backpropagates through time (BPTT) :
		-- 1. backward through feedback and input layers,
		-- 2. updates parameters
		model:backwardThroughTime()
		model:updateParameters(lr)
		model:zeroGradParameters()
	end
end

-- dummy dataset (task predict the next item)
dataset = torch.randn(nIndex, inputSize)

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
for i = 1, 10000 do
   --get a batch of inputs
   local inputs = dataset:index(1, offsets)
   -- shift of one batch indexes
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   -- a batch of targets
   local targets = dataset[{{},1}]:index(1,offsets)
   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end
