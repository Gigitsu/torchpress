batchSize = 1
rho = 5
hiddenSize = 10
nIndex = 10000

dataset = torch.randperm(nIndex)

offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

for i = 1, 1 do
   local inputs, targets = {}, {}
   for step = 1, rho do
      --get a batch of inputs
      print('input ')
      print(dataset:index(1, offsets))

      -- shift of one batch indexes to create the target
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      -- a batch of targets
      print('target ')
      print(dataset:index(1, offsets))
   end
end
