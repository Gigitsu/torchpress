require 'nn'

function gradientUpgrade(model, x, y, criterion, learningRate)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
	local gradOutputs = criterion:backward(prediction, y)
	model:zeroGradParameters()
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
end

model = nn.Sequential()
model:add(nn.Linear(5,1))

x1 = torch.rand(5)
y1 = torch.Tensor({1})
x2 = torch.rand(5)
y2 = torch.Tensor({-1})

criterion = nn.MarginCriterion(1)

for i = 1, 10000 do
	gradientUpgrade(model, x1, y1, criterion, 0.01)
	gradientUpgrade(model, x2, y2, criterion, 0.01)
end

-- with y1[1] we extract the first value in the tensor
print('prediction for x1 = ' .. model:forward(x1)[1] .. ' expected value ' .. y1[1])
print('prediction for x2 = ' .. model:forward(x2)[1] .. ' expected value ' .. y2[1])

print('loss after training for x1 = ' .. criterion:forward(model:forward(x1), y1))
print('loss after training for x2 = ' .. criterion:forward(model:forward(x2), y2))
