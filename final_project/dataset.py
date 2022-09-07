import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from numpy.random import seed, shuffle
from PIL import Image
import torch

# def denorm(tensor, device):
# 	tensor = tensor.to(device)
# 	std = torch.Tensor([.485, .456, .406]).reshape(-1,1,1).to(device)
# 	mean = torch.Tensor([.229, .224, .225]).reshape(-1,1,1).to(device)
# 	res = torch.clamp(tensor * std + mean, 0 , 1)
# 	return res

resize = transforms.Resize(512)
normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
# trans = transforms.Compose([resize,transforms.RandomCrop(512), transforms.ToTensor(), normalize])
trans = transforms.Compose([resize,transforms.RandomCrop(512), transforms.ToTensor()])
trans_style = transforms.Compose([resize,transforms.CenterCrop(512), transforms.ToTensor()])

class DataSet(Dataset):

	def __init__(self, image_dir, num=10000, transforms=trans, random_seed=1126):
		super().__init__()
		images = glob(os.path.join(image_dir, '*'))
		if random_seed != 1126: seed(random_seed)
		
		shuffle(images)
		images = images[:num]
		self.images = images
		self.transforms = transforms

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img = self.images[idx]
		img = Image.open(img)
		if self.transforms:
			img = self.transforms(img)
		return img

class StyleSet(Dataset):

	def __init__(self, image_dir, transforms=trans_style, random_seed=1126):
		super().__init__()
		images = glob(os.path.join(image_dir, '*'))
		#if random_seed != 1126: seed(random_seed)

		#shuffle(images)
		print(images)
		self.images = images
		self.transforms = transforms

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img = self.images[idx]
		img = Image.open(img)
		if self.transforms:
			img = self.transforms(img)
		return img