-- ===================================================================
-- Below variables should be defined in upper layer as global
--		; dTrainInput, dTrainLabel, dSet5Input, dSet5Label
-- ===================================================================

require 'etc'

function infer_one_image(model, criterion, img_input, img_label)

	local batchInputs = img_input.img:cuda()
	local img_label_shave = shave(img_label.img, img_label.size.height, img_label.size.width)
	local batchLabels = img_label_shave:cuda()

	local outputs = model:forward(batchInputs)
	print ( #outputs )
	print ( #batchLabels )
	local loss = criterion:forward(outputs, batchLabels)

	return output, loss

end

function test_full_image(model, criterion, dSetInput, dSetLabel)

	local AvgPsnr = 0
	local AvgMse = 0 
	for i=1, #(dSetInput.name) do
		local tmp_dSetInput = {}
		local tmp_dSetLabel = {}
		tmp_dSetInput.name = dSetInput.name[i]
		tmp_dSetInput.size = dSetInput.size[i]
		tmp_dSetInput.img  = dSetInput.img[i]
		tmp_dSetLabel.name = dSetLabel.name[i]
		tmp_dSetLabel.size = dSetLabel.size[i]
		tmp_dSetLabel.img  = dSetLabel.img[i]
		_, mse = infer_one_image(model, criterion, tmp_dSetInput, tmp_dSetLabel)
		AvgMse = AvgMse + mse
		AvgPsnr = AvgPsnr + 10*math.log10(1/mse)
	end

	AvgMse = AvgMse / size(dSetInput)[1]
	AvgPsnr = AvgPsnr / size(dSetInput)[1]

	return AvgPsnr, AvgMse
end



