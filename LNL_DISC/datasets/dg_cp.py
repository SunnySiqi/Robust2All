from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
import random
from PIL import Image, ImageFile
import torch
from .randaugment import TransformFixMatchLarge
import os
import numpy as np
import skimage.io
import pandas as pd
from .tps_transform import TPSTransform
ImageFile.LOAD_TRUNCATED_IMAGES = True

CP_dict = {
			"BRD-A29260609": 0,
			"BRD-K04185004": 1,
			"BRD-K21680192": 2,
			"DMSO": 3,
		}

class dg_cp_dataset(Dataset):

	def __init__(self, root_dir, mode, test_domain, train_domain):

		self.root = root_dir
		self.mode = mode
		self.test_domain = test_domain
		self.train_domain = train_domain
		self.train_labels = {}
		self.test_labels = {}
		self.domain_labels = {}
		self.metadata = pd.read_csv(self.root+'/cp.csv')
		CP_weak = T.Compose([
			T.Resize(256),
			T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
			T.RandomHorizontalFlip(),
			T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
					(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084))])


		CP_strong = T.Compose([
			T.Resize(256),
			T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
			T.RandomHorizontalFlip(),
			TPSTransform(p=1),
			T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
					(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084))])

		CP_test = T.Compose([
			T.Resize(256),
			T.CenterCrop(224),
			T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
					(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084))])
		
		self.transform_strong = CP_strong
		self.transform_weak = CP_weak
		self.transform_test = CP_test


		if mode == 'test' or mode == 'test_ind':
			for i in range(len(self.metadata)):
				if self.metadata.loc[i, "train_test_split"] == test_domain:
					self.test_labels[self.metadata.loc[i, "file_path"]] = CP_dict[self.metadata.loc[i, "label"]]  
			self.test_imgs = list(self.test_labels)
		
		else:
			for i in range(len(self.metadata)):
				if self.metadata.loc[i, "train_test_split"] in self.train_domain:
					self.train_labels[self.metadata.loc[i, "file_path"]] = CP_dict[self.metadata.loc[i, "label"]] 
					self.domain_labels[self.metadata.loc[i, "file_path"]] = self.metadata.loc[i, "train_test_split"] 
			self.train_imgs = list(self.train_labels)
			np.random.seed(1234)
			idx = np.random.permutation(len(self.train_imgs))
			self.train_imgs = np.array(self.train_imgs)[idx]
			val_perc = 20
			val_len = len(self.train_imgs)*val_perc//100                   
			if mode == 'val':
				self.val_imgs = self.train_imgs[:val_len]
			elif mode == 'train' or mode == 'all':
				self.train_imgs_split = self.train_imgs[val_len:]
				self.num_samples = len(self.train_imgs_split)
				self.index_dict = {}
				for i in range(len(self.train_imgs_split)):
					self.index_dict[self.train_imgs_split[i]] = i


	def sample_subset(self, num_class=4):  #sample a class-balanced subset
		random.shuffle(self.train_imgs_split)
		class_num = torch.zeros(num_class)
		self.train_imgs_subset = []
		for impath in self.train_imgs_split:
			label = self.train_labels[impath]
			if class_num[label] < int(self.num_samples / num_class):
				self.train_imgs_subset.append(impath)
				class_num[label] += 1
		random.shuffle(self.train_imgs_subset)
		return
	
	def load_img(self, img_path, aug='weak'):
		img = skimage.io.imread(img_path)
		img = np.reshape(img, (img.shape[0], 160, -1), order="F")
		img = T.ToTensor()(img)
		if self.mode == 'train' and aug == 'weak':
			img = self.transform_weak(img)
		elif self.mode == 'train' and aug == 'strong':
			img = self.transform_strong(img)
		elif self.mode == 'test':
			img = self.transform_test(img)
		return img

	def __getitem__(self, index):
		if self.mode == 'train' or self.mode == 'all':
			img_path = self.train_imgs_split[index]
			ind = self.index_dict[img_path]
			target = self.train_labels[img_path]
			img_path = '%s/images/'%self.root + self.train_imgs[index]
			w_img = self.load_img(img_path)
			s_img = self.load_img(img_path, aug='strong')
			return (w_img, s_img), target, ind
		elif self.mode == 'test':
			img_path = '%s/images/'%self.root + self.test_imgs[index]
			target = self.test_labels[self.test_imgs[index]]
			img = self.load_img(img_path)
			return img, target
		elif self.mode == 'test_ind':
			img_path = '%s/images/'%self.root + self.test_imgs[index]
			target = self.test_labels[self.test_imgs[index]]
			img = self.load_img(img_path)
			return img, target, img_path
		elif self.mode == 'val':
			img_path = self.val_imgs[index]
			target = self.train_labels[img_path]
			img_path = '%s/images/'%self.root + img_path
			img = self.load_img(img_path)
			return img, target

	def __len__(self):
		if self.mode == 'test' or self.mode == 'test_ind':
			return len(self.test_imgs)
		elif self.mode == 'val':
			return len(self.val_imgs)
		elif self.mode == 'train':
			#return len(self.train_imgs_subset)
			return len(self.train_imgs_split)
		elif self.mode == 'all':
			return len(self.train_imgs_split)


class dg_cp_dataloader():
	def __init__(self, root_dir, batch_size, num_workers, test_domain, train_domain):
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.root_dir = root_dir
		self.test_domain = test_domain
		self.train_domain = train_domain

	def run_all(self):
		train_dataset = dg_cp_dataset(root_dir=self.root_dir,
										 mode='all',
										 test_domain=self.test_domain,
										 train_domain=self.train_domain)
		train_loader = DataLoader(dataset=train_dataset,
								  batch_size=self.batch_size,
								  shuffle=True,
								  num_workers=self.num_workers,
								  pin_memory=True)

		test_dataset = dg_cp_dataset(root_dir=self.root_dir,
										mode='test',
										test_domain=self.test_domain,
										train_domain=self.train_domain)
		test_loader = DataLoader(dataset=test_dataset,
								 batch_size=self.batch_size,
								 shuffle=False,
								 num_workers=self.num_workers,
								 pin_memory=True)

		return train_loader, test_loader

	def run(self):
		train_dataset = dg_cp_dataset(root_dir=self.root_dir,
										 mode='train',
										 test_domain=self.test_domain,
										 train_domain=self.train_domain)
		#train_dataset.sample_subset()
		train_loader = DataLoader(dataset=train_dataset,
								  batch_size=self.batch_size,
								  shuffle=True,
								  num_workers=self.num_workers,
								  pin_memory=True)
		print("TRAIN LOADER LENGTH!!!!!", len(train_dataset))
		test_dataset = dg_cp_dataset(root_dir=self.root_dir,
										mode='test',
										test_domain=self.test_domain,
										train_domain=self.train_domain)
		test_loader = DataLoader(dataset=test_dataset,
								 batch_size=self.batch_size,
								 shuffle=False,
								 num_workers=self.num_workers,
								 pin_memory=True)
		print("TEST LOADER LENGTH!!!!!", len(test_dataset))
		eval_dataset = dg_cp_dataset(root_dir=self.root_dir,
										mode='val',
										test_domain=self.test_domain,
										train_domain=self.train_domain)
		print("EVAL LOADER LENGTH!!!!!", len(eval_dataset))
		eval_loader = DataLoader(dataset=eval_dataset,
								 batch_size=self.batch_size,
								 shuffle=False,
								 num_workers=self.num_workers,
								 pin_memory=True)

		return train_loader, eval_loader, test_loader

