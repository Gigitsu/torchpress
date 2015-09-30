require 'rnn'
require 'optim'
require('mobdebug').start()

batchSize = 50
rho = 10
hiddenSize = 64
inputSize = 4
outputSize = 1

seriesSize = 10000
seriesEval = 1000

model = nn.Sequential()
model:add(nn.FastLSTM(inputSize, hiddenSize, rho))
--model:add(nn.Linear(inputSize, hiddenSize))
--model:add(nn.Tanh())
model:add(nn.Linear(hiddenSize, outputSize))

criterion = nn.MSECriterion()


-- dummy dataset (task predict the next item)
dataset = torch.randn(seriesSize, inputSize)
evalset = torch.randn(seriesEval, inputSize)

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

-- method to compute a batch
function nextBatch()
	--get a batch of inputs
	local inputs = dataset:index(1, offsets)
	-- shift of one batch indexes
	offsets:add(1)
	for j=1,batchSize do
		if offsets[j] > seriesSize then
			offsets[j] = 1
		end
	end
	-- a batch of targets
	local targets = dataset[{{},{outputSize}}]:index(1,offsets)
	return inputs, targets
end

-- get weights and loss wrt weights from the model
x, dl_dx = model:getParameters()
updateInterval = rho - 1

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

--function for validation
validate = function(data)
  model:evaluate()
  local maxPosition = data:size()[1] - 1
  local cumulatedError = 0
  for i = 1, maxPosition do
    model:forget()
		local x = data[i]
		local y = torch.DoubleTensor{data[i+1][4]}
      local prediction = model:forward(x)
      local err = criterion:forward(prediction, y)
      cumulatedError = cumulatedError + err
	end
   model:training()
	return cumulatedError / maxPosition
end

sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

lr = 0.1
for i = 1, 10e4 do
	-- train a mini_batch of batchSize in parallel
	_, fs = optim.sgd(feval,x, sgd_params)


   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      model:backwardThroughTime()
   end

	if sgd_params.evalCounter % 100 == 0 then
		print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
		local validationError = validate(evalset)
		print('error on validation ' .. validationError)
	end
end
