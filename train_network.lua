-- ===================================================================
-- Below variables should be defined in upper layer as global
--		; dTrainInput, dTrainLabel, dSet5Input, dSet5Label
-- ===================================================================

-- My libraries
require 'test_network'

local function one_train_iter (iter, shuffled_index, AvgPsnr, AvgMse, dTrainInput, dTrainLabel )
	local batchInputs = dTrainInput:index(1, shuffled_index[{ {batch_size*(iter-1)+1, batch_size*iter} }]):cuda()
	local batchLabels = dTrainLabel:index(1, shuffled_index[{ {batch_size*(iter-1)+1, batch_size*iter} }]):cuda()
	batchInputs = batchInputs:cuda()
	batchLabels = batchLabels:cuda()

	local function feval(params)
		gradParams:zero()
	
		local outputs = model:forward(batchInputs)
		local loss = criterion:forward(outputs, batchLabels)
		local dloss_doutput = criterion:backward(outputs, batchLabels)
		model:backward(batchInputs, dloss_doutput)
	
		return loss,gradParams
	end

	local _, loss = optimMethod(feval, params, optimState)
	AvgPsnr = AvgPsnr + 10*math.log10(1/loss[1])
	AvgMse = AvgMse + loss[1]

	return AvgPsnr, AvgMse
end

local function one_train_epoch (epoch, dTrainInput, dTrainLabel)
	-- Batch Shuffle
	local shuffled_index = torch.randperm( dTrainInput:size()[1] ):long()
	local time = sys.clock()
	local AvgPsnr = 0
	local AvgMse = 0

	for iter=1,max_iter do
		AvgPsnr, AvgMse = one_train_iter(iter, shuffled_index, AvgPsnr, AvgMse, dTrainInput, dTrainLabel)
	end
	AvgPsnr = AvgPsnr / max_iter
	AvgMse = AvgMse / max_iter
	print( 'epoch : '.. epoch, 'psnr : ' .. AvgPsnr, 'mse : ' .. AvgMse, 'Time cost : ' .. sys.clock() - time)

	return 1
end

function train_network(model, criterion, dTrainInput, dTrainLabel, dSet5Input, dSet5Label, dSet14Input, dSet14Label)

	params, gradParams = model:getParameters()
	max_iter = math.floor( dTrainInput:size()[1]/batch_size )
	print ( 'iter per epoch : ' .. max_iter, 'batch size : ' .. batch_size )

	-- Training
	for epoch=1,max_epoch do
		one_train_epoch(epoch, dTrainInput, dTrainLabel)
		local AvgPsnr_Set5 = 0
		local AvgMse_set5 = 0
		local AvgPsnr_Set14 = 0
		local AvgMse_set14 = 0

		if epoch%1 == 0 then
			if dSet5Input ~= nil then
				AvgPsnr_Set5, AvgMse_Set5 = test_full_image(model, criterion, dSet5Input, dSet5Label)
				print(AvgPsnr_Set5)
				print('Set5 psnr : ' .. tostring(AvgPsnr_Set5) .. ' mse : ' .. tostring(AvgMse_Set5) ) 
			end
			-- if dSet14Input ~= nil then
			-- 	AvgPsnr_Set14, AvgMse_Set14 = test_full_image(model, criterion, dSet14Input, dSet14Label)
			-- 	print('Set14 psnr : ' .. AvgPsnr_Set14 .. ' mse : ' .. AvgMse_Set14) 
			-- end
		end
	end
	params = nil
	gradParams =  nil
	return 1
end


