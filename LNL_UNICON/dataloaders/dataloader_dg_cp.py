from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image, ImageFile
import skimage.io
import pandas as pd
import json
import torch
import os
import pickle
from autoaugment import CIFAR10Policy, ImageNetPolicy
from tps_transform import TPSTransform
ImageFile.LOAD_TRUNCATED_IMAGES = True

CP_dict = {
			"BRD-A29260609": 0,
			"BRD-K04185004": 1,
			"BRD-K21680192": 2,
			"DMSO": 3,
		}

class dg_CP_dataset(Dataset):
	def __init__(
		self, sample_ratio,
		root,
		transform,
		mode,
		test_domain,
		train_domain,
		save_file,
		pred=[],
		probability=[],
		paths=[],
		num_class=4,
		use_domainlabels=False
	):

		self.root = root
		self.transform = transform
		self.mode = mode
		self.test_domain = test_domain
		self.train_domain = train_domain
		self.train_labels = {}
		self.test_labels  = {}
		self.domain_labels = {}
		self.metadata = pd.read_csv(self.root+'/cp.csv')

		if mode == 'test':
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
			idx = np.random.permutation(len(self.train_imgs ))
			self.train_imgs = np.array(self.train_imgs)[idx]

			val_perc = 20
			val_len = len(self.train_imgs)*val_perc//100                   
			if mode == 'val':
				self.val_imgs = self.train_imgs[:val_len]
			else:
				self.train_imgs_split = self.train_imgs[val_len:]
				self.mode_train_imgs = []
				class_num = torch.zeros(num_class)
				if mode == 'all':
					self.mode_train_imgs = self.train_imgs_split
					random.shuffle(self.mode_train_imgs)

				elif mode == 'labeled' and not use_domainlabels:
					class_inds = {}
					prob_class_domain_dict = {}
					for kk in range(num_class):
						class_inds[kk] = {}
						prob_class_domain_dict[kk] = {}
						for d in list(set(self.domain_labels.values())):
							class_inds[kk][d] = [i for i,x in enumerate(self.train_imgs_split) if self.train_labels[x]==kk and self.domain_labels[x] == d]
							prob_class_domain_dict[kk][d] = probability[class_inds[kk][d]]
 
					train_imgs = self.train_imgs_split
					num_samples = len(train_imgs)
					pred_idx   = np.zeros(int(sample_ratio*num_samples))
					class_len  = int(sample_ratio*num_samples/num_class)
					size_pred  = 0
					class_ind  = {}

					## Get the class indices
					for kk in range(num_class):
						class_ind[kk] = [i for i,x in enumerate(train_imgs) if self.train_labels[x]==kk]

					## Creating the Class Balance
					for i in range(num_class):
						sorted_indices  = np.argsort(probability[class_ind[i]])      ##  Sorted indices for each class  
						class_indices   = np.array(class_ind[i])                     ##  Class indices  
						size1 = len(class_indices)
						try:
							pred_idx[size_pred:size_pred+class_len] = class_indices[sorted_indices[0:class_len].cpu().numpy()].squeeze()
							size_pred += class_len
						except:
							pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
							size_pred += size1
					## Predicted Clean Samples  
					pred_idx = [int(x) for x in list(pred_idx)]
					np.savez(save_file, index = pred_idx)
					self.mode_train_imgs  = [train_imgs[i] for i in pred_idx]
					probability[probability<0.5] = 0                        ## Weight Adjustment 
					self.probability = [1-probability[i] for i in pred_idx]
					print("%s data has a size of %d"%(self.mode,len(self.mode_train_imgs)))
				
				elif mode == 'labeled' and use_domainlabels:
					train_imgs = self.train_imgs_split
					num_samples = len(train_imgs)
					pred_idx = []
					#class_len_per_d  = int(sample_ratio*num_samples/(num_class*len(train_domain)))
					#print("Class_len_per_d!!!!!!!!!", class_len_per_d)
					size_pred  = 0
					class_ind  = {}

					for d in self.train_domain: 
						#pred_idx[d]   = np.zeros(int(sample_ratio*num_samples/len(train_domain)))
						## Get the class indices
						for kk in range(num_class):
							class_ind[kk] = [i for i,x in enumerate(train_imgs) if self.train_labels[x]==kk and self.domain_labels[x] == d]
						#print("class_ind!!!!!!!!!!", class_ind)

						## Creating the Class Balance
						for i in range(num_class):
							sorted_indices  = np.argsort(probability[class_ind[i]])      ##  Sorted indices for each class  
							class_indices   = np.array(class_ind[i])                     ##  Class indices  
							size1 = len(class_indices)

							d_i_prob = probability[class_ind[i]].cpu().numpy()
							threshold   = np.mean(d_i_prob)                                           ## Simply Take the average as the threshold
							#print("Threshold!!!!!", threshold)
							if threshold>0.7: #args.du
								threshold = threshold - (threshold-np.min(d_i_prob))/5.0  #args.tau
							sample_ratio = np.sum(d_i_prob<threshold)/size1     
							class_len_per_d = int(size1*sample_ratio)
							try:
								pred_idx[size_pred:size_pred+class_len_per_d] = class_indices[sorted_indices[0:class_len_per_d].cpu().numpy()].squeeze()
								size_pred += class_len_per_d
							except:
								pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
								size_pred += size1
					
					## Predicted Clean Samples  
					pred_idx = [int(x) for x in list(pred_idx)]
					np.savez(save_file, index = pred_idx)
					self.mode_train_imgs  = [train_imgs[i] for i in pred_idx]
					probability[probability<0.5] = 0                        ## Weight Adjustment 
					self.probability = [1-probability[i] for i in pred_idx]
					print("%s data has a size of %d"%(self.mode,len(self.mode_train_imgs)))

				elif self.mode == "unlabeled":
					train_imgs = self.train_imgs_split
					num_samples = len(train_imgs)
					pred_idx1 = np.load(save_file)['index']
					idx = list(range(num_samples))
					pred_idx = [x for x in idx if x not in pred_idx1] 
					self.mode_train_imgs = [train_imgs[i] for i in pred_idx]                         
					print("%s data has a size of %d"%(self.mode,len(self.mode_train_imgs)))


	def __getitem__(self, index):
		if self.mode == "labeled":
			img_path = self.mode_train_imgs[index]
			target = self.train_labels[img_path]
			prob = self.probability[index]
			path = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/images/'+img_path
			image = self.load_img(path)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)
			
			return img1, img2, img3, img4, target, prob

		elif self.mode == "unlabeled":
			img_path = self.mode_train_imgs[index]
			path = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/images/'+img_path
			image = self.load_img(path)
			img1 = self.transform[0](image)
			img2 = self.transform[1](image)
			img3 = self.transform[2](image)
			img4 = self.transform[3](image)

			return img1, img2, img3, img4
			
		elif self.mode == "all":
			img_path = self.mode_train_imgs[index]
			target = self.train_labels[img_path]
			path = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/images/'+img_path
			img = self.load_img(path)
			return img, target, img_path

		elif self.mode == "test":
			img_path = self.test_imgs[index]
			target = self.test_labels[img_path]
			path = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/images/'+img_path
			image = self.load_img(path)
			img = self.transform(image)
			return img, target
			
		elif self.mode == "val":
			img_path = self.val_imgs[index]
			target = self.train_labels[img_path]
			path = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/images/'+img_path
			image = self.load_img(path)
			img = self.transform(image)
			return img, target

	def load_img(self, img_path):
		img = skimage.io.imread(img_path)
		img = np.reshape(img, (img.shape[0], 160, -1), order="F")
		img = T.ToTensor()(img)
		return img

	def __len__(self):
		if self.mode == "test":
			return len(self.test_imgs)
		if self.mode == "val":
			return len(self.val_imgs)
		else:
			return len(self.mode_train_imgs)


class dg_CP_dataloader:
	def __init__(
		self,
		root,
		batch_size,
		warmup_batch_size,
		num_workers,
		test_domain,
		train_domain,
		save_file,
		use_domainlabels):

		self.batch_size = batch_size
		self.warmup_batch_size = warmup_batch_size
		self.num_workers = num_workers
		self.root = root
		self.train_domain = train_domain
		self.test_domain = test_domain
		self.save_file = save_file
		self.use_domainlabels = use_domainlabels

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

		self.transforms = {
			"warmup": CP_weak,
			"unlabeled": [
						CP_strong,
						CP_strong,
						CP_weak,
						CP_weak
					],
			"labeled": [
						CP_strong,
						CP_strong,
						CP_weak,
						CP_weak
					]
		}
		self.transforms_val = CP_test

		self.transforms_test = CP_test

	def run(self, sample_r, mode,  pred=[], prob=[], paths=[]):
		if mode == "warmup":
			warmup_dataset = dg_CP_dataset(sample_r,
				self.root,
				transform=self.transforms["warmup"],
				mode="all",
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				use_domainlabels=self.use_domainlabels
			)
			warmup_loader = DataLoader(
				dataset=warmup_dataset,
				batch_size=self.warmup_batch_size*2,
				shuffle=True,
				num_workers=self.num_workers,
			)
			return warmup_loader
		elif mode == "train":
			labeled_dataset = dg_CP_dataset( sample_r,
				self.root, 
				transform=self.transforms["labeled"],
				mode="labeled", 
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				pred=pred,
				probability=prob,
				paths=paths,
				use_domainlabels=self.use_domainlabels
			)
			labeled_loader = DataLoader(
				dataset=labeled_dataset,
				batch_size=self.batch_size,
				shuffle=True, 
				num_workers=self.num_workers, drop_last = True
			)
			unlabeled_dataset = dg_CP_dataset(sample_r,
				self.root, 
				transform = self.transforms["unlabeled"],
				mode = "unlabeled",
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				pred = pred,
				probability=prob,
				paths=paths,
				use_domainlabels=self.use_domainlabels
			)
			unlabeled_loader = DataLoader(
				dataset=unlabeled_dataset,
				batch_size=self.batch_size,
				shuffle=True, 
				num_workers=self.num_workers, drop_last = True
			)
			return labeled_loader, unlabeled_loader

		elif mode == "eval_train":
			eval_dataset = dg_CP_dataset( sample_r,
				self.root, 
				transform=self.transforms_test,
				mode="all",
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				use_domainlabels=self.use_domainlabels
			)
			eval_loader = DataLoader(
				dataset=eval_dataset,
				batch_size=self.batch_size*4,
				shuffle=False,
				num_workers=self.num_workers,
			)
			return eval_loader

		elif mode == "test":
			test_dataset = dg_CP_dataset(
				sample_r,self.root,  transform=self.transforms_test, mode="test",
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				use_domainlabels=self.use_domainlabels
			)
			test_loader = DataLoader(
				dataset=test_dataset,
				batch_size=self.batch_size,
				shuffle=False,
				num_workers=self.num_workers,
			)
			return test_loader

		elif mode == "val":
			val_dataset = dg_CP_dataset(
				sample_r, self.root ,transform=self.transforms_val, mode="val",
				test_domain=self.test_domain,
				train_domain=self.train_domain,
				save_file=self.save_file,
				use_domainlabels=self.use_domainlabels
			)
			val_loader = DataLoader(
				dataset=val_dataset,
				batch_size=self.batch_size,
				shuffle=False,
				num_workers=self.num_workers,
			)
			return val_loader
