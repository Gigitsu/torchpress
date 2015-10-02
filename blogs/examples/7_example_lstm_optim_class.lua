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
