
path_train_data  = "./gen_patch/patches/patch_test_data.bin"
path_train_label = "./gen_patch/patches/patch_test_label.bin"
path_Set5_data   = "./gen_patch/patches/Set5/input/"
path_Set5_label  = "./gen_patch/patches/Set5/label/"
path_Set14_data   = "./gen_patch/patches/Set14/input/"
path_Set14_label  = "./gen_patch/patches/Set14/label/"

SizeGap = 12

imgsizeTrainInput = {}
imgsizeTrainInput.width  = 33
imgsizeTrainInput.height = 33
imgsizeTrainLabel = {}
imgsizeTrainLabel.width  = imgsizeTrainInput.width  - SizeGap
imgsizeTrainLabel.height = imgsizeTrainInput.height - SizeGap

batch_size = 32
max_epoch = 100
optimMethod = optim.sgd
optimState = {learningRate=1e-5}

