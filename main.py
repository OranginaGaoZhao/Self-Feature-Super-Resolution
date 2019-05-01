import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from PIL import Image
import numpy as np
from model import Net

from tqdm import tqdm
from os import listdir
from os.path import join, basename

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg",".bmp",".jpeg",".png",".JPG"])

def main():
	model = Net()

	model_name = 'orange_sr_model_epoch_200.pth'
	model.load_state_dict(torch.load(model_name, map_location = lambda storage, loc: storage))


	test_path = 'test_images'
	image_files = [join(test_path, x) for x in listdir(test_path) if is_image_file(x)]
	

	for i, file in tqdm(enumerate(image_files)):

		input_image = file

		file_name = basename(file)
		img = Image.open(input_image).convert('RGB')
		img = img.resize((3*img.size[0], 3*img.size[1]), Image.BILINEAR)
		im_input = Variable(ToTensor()(img)).view(1,-1,img.size[1],img.size[0]) 
		_, _, out = model(im_input)

		out_img = out.data[0].numpy()
		out_img *= 255.0
		out_img = out_img.clip(0,255)

		out_img_r = Image.fromarray(np.uint8(out_img[0]),mode='L')
		out_img_g = Image.fromarray(np.uint8(out_img[1]),mode='L')
		out_img_b = Image.fromarray(np.uint8(out_img[2]),mode='L')

		sr_img = [out_img_r, out_img_g, out_img_b]
		sr_img = Image.merge('RGB',sr_img)

		sr_img.save('new_sr_'+file_name+'.png')

if __name__ == '__main__':
	main()
