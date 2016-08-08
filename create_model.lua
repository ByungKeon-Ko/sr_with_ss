require 'torch'
require 'cudnn'

local nn = require 'nn'
local nninit = require 'nninit'

torch.setdefaulttensortype('torch.FloatTensor')

function create_model ()
	print('Start to Create Network Model' )
	-- feature map depth
	local nInChannel = 1
	local nOutChannel = 1
	local nDepth1 = 64
	local nDepth2 = 32
	-- filter size
	local f1 = 9
	local f2 = 1
	local f3 = 5

	local model = nn.Sequential()
	-- 1st layer : Patch Extraction and representation
	model:add(nn.SpatialConvolution(nInChannel, nDepth1, f1, f1 )
		:init('weight', nninit.normal, 0, 1e-3)
		:init('bias', nninit.constant, 0) )
	model:add(nn.ReLU())

	-- 2nd layer : Non-linear mapping
	model:add(nn.SpatialConvolution(nDepth1, nDepth2, f2, f2 )
		:init('weight', nninit.normal, 0, 1e-3)
		:init('bias', nninit.constant, 0) )
	model:add(nn.ReLU())

	-- 3rd layer : Reconstruction
	model:add(nn.SpatialConvolution(nDepth2, nOutChannel, f3, f3 )
		:init('weight', nninit.normal, 0, 1e-3)
		:init('bias', nninit.constant, 0) )
	model:add(nn.ReLU())
	criterion = nn.MSECriterion()

	return model:cuda(), criterion:cuda()
end




