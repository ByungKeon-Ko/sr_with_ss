require 'image';
require 'etc';

local function load_one_patch ( bin_file, size_patch )
	local loaded_patch = torch.Tensor(bin_file)
	local idx_x = 1;
	local idx_y = 1;
	local cnt = 1;

	loaded_patch = loaded_patch/255.0
	if (#bin_file ~= size_patch.height * size_patch.width) then
		print("Size mismatch!!" )
		print("#bin_file : ", #bin_file)
		print("size_patch : ", size_patch.width, size_patch.height )
	end

	loaded_patch = torch.reshape(loaded_patch, 1, 1, size_patch.width, size_patch.height)
	loaded_patch = loaded_patch:permute(1, 2, 4,3)

	return loaded_patch
end

local function convert_stream_to_img_array ( byte_stream, imgsize )
	local one_stream_size = imgsize.width * imgsize.height
	local num_patch = math.floor( #byte_stream / one_stream_size )
	-- local patch_array = torch.ByteTensor( byte_stream )
	local patch_array = torch.Tensor( byte_stream )
	patch_array = patch_array / 255.0
	patch_array = torch.reshape(patch_array, num_patch, 1, imgsize.width, imgsize.height )
	patch_array = patch_array:permute( 1,2,4,3)
	return patch_array
end

local function load_byte_stream(path_bin, dtype)

	local file = assert(io.open(path_bin, "rb" ))  -- 174 x 195 pxl
	local block = nil
	if dtype == "uint8" then
		block = 1
	else
		print("dtype Error!!, only uint8 type byte stream is supported!!")
		os.exit()
	end

	local byte_stream = {}
	
	local loop_cnt = 1
	while true do
		local byte = file:read(block)
		if byte == nil then
			break
		else
			byte_stream[#byte_stream+1] = string.byte(byte)
			if loop_cnt%10000000 == 0 then
				print( "Read byte stream, ", loop_cnt)
			end
			loop_cnt = loop_cnt + 1
		end
	end
	file:close()
	return byte_stream
end

function load_patch_array(path_bin, imgsize)
	-- local imgsize = {}
	-- imgsize.width  = 33
	-- imgsize.height = 33

	local byte_stream = load_byte_stream(path_bin, "uint8")
	local patch_array = convert_stream_to_img_array ( byte_stream, imgsize )
	return patch_array
end

-- Pick one image from array randomly for debugging utility
function random_patch_from_array ( patch_array )
	local rand_index = math.random( (#patch_array)[1] )
	return patch_array[ { {rand_index}, {}, {},{} }]:reshape(1, imgsize.height, imgsize.width )
end

function load_one_image ( path_bin, imgsize )
	
	local byte_stream = load_byte_stream(path_bin, "uint8")
	-- load data patch by patch
	local loaded_image = load_one_patch(byte_stream, imgsize)
	return loaded_image

end

function load_test_img_array ( path_folder )
	local file_list = scandir(path_folder)
	local test_img_array = {}
	test_img_array.name = {}
	test_img_array.img = {}
	test_img_array.size = {}

	for i=3, #file_list do
		-- File name example
		--	: input_baby_510_510.bin
		--	: label_woman_342_228.bin
		--	; <type>_<imgname>_<hei>_<wid>.bin
		local decomposed_path = string.split(file_list[i], "_")
		local imgtype = decomposed_path[1]
		local imgname = decomposed_path[2]
		local imgsize = {}
		imgsize.height = tonumber(decomposed_path[3])
		imgsize.width  = tonumber( string.split(decomposed_path[4], '%.')[1]   )
		test_img_array.img[i-2] = load_one_image( path_folder .. file_list[i], imgsize )
		test_img_array.name[i-2] = imgname
		test_img_array.size[i-2] = imgsize
	end

	return test_img_array
end

--	path_bin = "./gen_patch/patches/patch_test_data.bin"
--	imgsize = {}
--	imgsize.width  = 33
--	imgsize.height = 33
--	--	path_bin = "./gen_patch/patches/b1.bin"
--	--	imgsize = {}
--	--	imgsize.width  = 195
--	--	imgsize.height = 174
--	
--	patch_array = load_patch_array(path_bin)
--	
--	print("Load each patch start")
--	print(#patch_array)
--	
--	tmp_img = random_patch_from_array(patch_array)




