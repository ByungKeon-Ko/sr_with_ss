-- ===================================================================
-- To Do List ..
--	1. test_full_image : Compute Avg Psnr ( Should be able to be run not only in main.lua but also alone )
--  2. infer_full_image : Infer one full image ( using different file from main.lua )
--	3. different learning rate depends on layer
--	4. Image pre-precessing? optim method?, other options for fast learning?? or fast torch?

-- ===================================================================

-- Tensorflow basic libraries
require 'nn';
require 'cutorch';
require 'torch';
require 'optim';
require 'cunn';
require 'cudnn';

-- Lua libraries
require 'image';

-- My libraries
require 'load_image';
require 'create_model';
require 'train_network';

-- Parameters
dofile 'parameters.lua'

-- Prepare data
-- 		Load Train patches
--		- data form : 4 dim ; { num_patch, nChannel, height, width }
print("start to load dataset")
dTrainInput = load_patch_array(path_train_data , imgsizeTrainInput)
collectgarbage()
dTrainLabel = load_patch_array(path_train_label, imgsizeTrainLabel)
collectgarbage()

-- 		Load Test images
dSet5Input  = load_test_img_array(path_Set5_data )
collectgarbage()
dSet5Label  = load_test_img_array(path_Set5_label)
collectgarbage()
dSet14Input  = load_test_img_array(path_Set14_data )
collectgarbage()
dSet14Label  = load_test_img_array(path_Set14_label)
collectgarbage()

-- Create Network model
print("start to create network model")
model, criterion = create_model( )
collectgarbage()

-- -- Traning
print("start to train")
train_network(model, criterion, dTrainInput, dTrainLabel, dSet5Input, dSet5Label, dSet14Input, dSet15Label)
collectgarbage()

print("Training Done!!")

