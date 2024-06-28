import copy
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import random
from PIL import Image
import pickle
import pandas as pd
import torchvision.transforms as transforms
import skimage.io

# TERRAINCOGNITA_PATH
# DOMAINNET_PATH
# OFFICEHOME_PATH
# PACS_PATH
# VLCS_PATH
# CLOTHING1M_PATH
# FASHIONMNIST_PATH 
# CP_PATH

class MyDataset(datasets.ImageFolder):
	def __init__(self, root, transform):
		super(MyDataset, self).__init__(root = root, transform=transform)
	def __getitem__(self, index):
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return sample, target, index

def _construct_dataset_helper(args, dataset_dict, train_transform, val_transform):
    train_datasets = []
    test_datasets = []
    train_dataset_lengths = 0
    val_dataset = None


    for d in args.training_data:
        train_datasets.append(dataset_dict[d])
        train_dataset_lengths += len(dataset_dict[d])

    for d in args.validation_data:
        test_datasets.append(dataset_dict[d])

    if args.train_val_split > 0:
        datasets_split_train = []
        datasets_split_val = []
        for d in train_datasets:
            lengths = [int(len(d) * args.train_val_split)]
            lengths.append(len(d) - lengths[0])
            train_split, val_split = torch.utils.data.random_split(
                d, lengths, torch.Generator().manual_seed(42)
            )
            train_split.dataset.transform = train_transform
            datasets_split_train.append(train_split)
            val_split.dataset.transform = val_transform
            datasets_split_val.append(val_split)
            # print("Train Split Type", type(train_split))
            # print("Val Split Type", type(val_split))
        train_datasets = datasets_split_train
        val_dataset = datasets_split_val

    return train_datasets, val_dataset, test_datasets, train_dataset_lengths

# def _construct_dataset_helper(args, dataset_dict, train_transform, val_transform):
# 	train_datasets = []
# 	val_datasets = []
# 	train_dataset_lengths = 0
# 	test_dataset = None


# 	for d in args.training_data:
# 		# if args.noise_ratio > 0:
# 		#     dataset_dict[d] = add_noise_dataset_dict(args, d, dataset_dict[d])
# 		train_datasets.append(dataset_dict[d])
# 		train_dataset_lengths += len(dataset_dict[d])

# 	for d in args.validation_data:
# 		val_datasets.append(dataset_dict[d])

# 	if args.train_val_split > 0:
# 		datasets_split_train = []
# 		datasets_split_val = []
# 		for d in train_datasets:
# 			lengths = [int(len(d) * args.train_val_split)]
# 			lengths.append(len(d) - lengths[0])
# 			train_split, val_split = torch.utils.data.random_split(
# 				d, lengths, torch.Generator().manual_seed(42)
# 			)
# 			train_split.dataset = copy.copy(d)
# 			train_split.dataset.transform = train_transform
# 			datasets_split_train.append(train_split)
# 		for idx, d in enumerate(val_datasets):
# 			lengths = [int(len(d) * args.train_val_split)]
# 			lengths.append(len(d) - lengths[0])
# 			train_split, val_split = torch.utils.data.random_split(
# 				 d, lengths, torch.Generator().manual_seed(42)
# 			)
# 			val_split.dataset.transform = val_transform
# 			datasets_split_val.append(val_split)

# 		train_datasets = datasets_split_train
# 		test_dataset = datasets_split_val

# 	else:

# 		test_dataset = val_datasets

# 	return train_datasets, test_dataset, train_dataset_lengths


def construct_dataset(args, train_transform, val_transform):

	if args.dataset == "domainnet":

		num_classes = 345

		transform_dict = {
			"sketch": train_transform,
			"real": train_transform,
			"clipart": train_transform,
			"infograph": train_transform,
			"quickdraw": train_transform,
			"painting": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		sketch_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "sketch"),
			transform=transform_dict["sketch"],
		)
		real_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "real"),
			transform=transform_dict["real"],
		)
		clipart_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "clipart"),
			transform=transform_dict["clipart"],
		)
		infograph_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "infograph"),
			transform=transform_dict["infograph"],
		)
		quickdraw_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "quickdraw"),
			transform=transform_dict["quickdraw"],
		)
		painting_dataset = MyDataset(
			os.path.join(DOMAINNET_PATH, "painting"),
			transform=transform_dict["painting"],
		)

		dataset_dict = {
			"sketch": sketch_dataset,
			"real": real_dataset,
			"clipart": clipart_dataset,
			"infograph": infograph_dataset,
			"quickdraw": quickdraw_dataset,
			"painting": painting_dataset,
		}

	elif args.dataset == "terraincognita":

		num_classes = 10

		transform_dict = {
			"location_100": train_transform,
			"location_38": train_transform,
			"location_43": train_transform,
			"location_46": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		location_100_dataset = MyDataset(
			os.path.join(TERRAINCOGNITA_PATH, "location_100"),
			transform=transform_dict["location_100"],
		)
		location_38_dataset = MyDataset(
			os.path.join(TERRAINCOGNITA_PATH, "location_38"),
			transform=transform_dict["location_38"],
		)
		location_43_dataset = MyDataset(
			os.path.join(TERRAINCOGNITA_PATH, "location_43"),
			transform=transform_dict["location_43"],
		)
		location_46_dataset = MyDataset(
			os.path.join(TERRAINCOGNITA_PATH, "location_46"),
			transform=transform_dict["location_46"],
		)

		dataset_dict = {
			"location_100": location_100_dataset,
			"location_38": location_38_dataset,
			"location_43": location_43_dataset,
			"location_46": location_46_dataset,
		}

	elif args.dataset == "officehome":

		num_classes = 65

		transform_dict = {
			"art": train_transform,
			"clipart": train_transform,
			"product": train_transform,
			"real": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		art_dataset = MyDataset(
			os.path.join(OFFICEHOME_PATH, "Art"),
			transform=transform_dict["art"],
		)
		clipart_dataset = MyDataset(
			os.path.join(OFFICEHOME_PATH, "Clipart"),
			transform=transform_dict["clipart"],
		)
		product_dataset = MyDataset(
			os.path.join(OFFICEHOME_PATH, "Product"),
			transform=transform_dict["product"],
		)
		real_dataset = MyDataset(
			os.path.join(OFFICEHOME_PATH, "Real"),
			transform=transform_dict["real"],
		)

		dataset_dict = {
			"art": art_dataset,
			"clipart": clipart_dataset,
			"product": product_dataset,
			"real": real_dataset,
		}

	elif args.dataset == "pacs":

		num_classes = 7

		transform_dict = {
			"art_painting": train_transform,
			"cartoon": train_transform,
			"photo": train_transform,
			"sketch": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		art_painting_dataset = MyDataset(
			os.path.join(PACS_PATH, "art_painting"),
			transform=transform_dict["art_painting"],
		)
		cartoon_dataset = MyDataset(
			os.path.join(PACS_PATH, "cartoon"),
			transform=transform_dict["cartoon"],
		)
		photo_dataset = MyDataset(
			os.path.join(PACS_PATH, "photo"),
			transform=transform_dict["photo"],
		)
		sketch_dataset = MyDataset(
			os.path.join(PACS_PATH, "sketch"),
			transform=transform_dict["sketch"],
		)

		dataset_dict = {
			"art_painting": art_painting_dataset,
			"cartoon": cartoon_dataset,
			"photo": photo_dataset,
			"sketch": sketch_dataset,
		}

	elif args.dataset == "vlcs":

		num_classes = 5

		transform_dict = {
			"caltech101": train_transform,
			"labelme": train_transform,
			"sun09": train_transform,
			"voc2007": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		caltech101_dataset = MyDataset(
			os.path.join(VLCS_PATH, "Caltech101"),
			transform=transform_dict["caltech101"],
		)
		labelme_dataset = MyDataset(
			os.path.join(VLCS_PATH, "LabelMe"),
			transform=transform_dict["labelme"],
		)
		sun09_dataset = MyDataset(
			os.path.join(VLCS_PATH, "SUN09"),
			transform=transform_dict["sun09"],
		)
		voc2007_dataset = MyDataset(
			os.path.join(VLCS_PATH, "VOC2007"),
			transform=transform_dict["voc2007"],
		)

		dataset_dict = {
			"caltech101": caltech101_dataset,
			"labelme": labelme_dataset,
			"sun09": sun09_dataset,
			"voc2007": voc2007_dataset,
		}

	elif args.dataset == "wilds_fmow":

		num_classes = 62

		train_datasets = []
		val_datasets = []
		train_dataset_lengths = []
		test_dataset = None

		datasets_list = get_fmow(train_transform, val_transform, args.validation_data)

		region0_dataset = datasets_list[0]
		region1_dataset = datasets_list[1]
		region2_dataset = datasets_list[2]
		region3_dataset = datasets_list[3]
		region4_dataset = datasets_list[4]
		region5_dataset = datasets_list[5]

		dataset_dict = {
			"region0": region0_dataset,
			"region1": region1_dataset,
			"region2": region2_dataset,
			"region3": region3_dataset,
			"region4": region4_dataset,
			"region5": region5_dataset,
		}
	
	elif args.dataset == 'clothing':
		num_classes = 5

		transform_dict = {
			"clothing1M": train_transform,
			"fashion-MNIST": val_transform
		}

		clothing1M_train = clothing_dataset(CLOTHING1M_PATH, transform_dict["clothing1M"],mode='train')
		clothing1M_val = clothing_dataset(CLOTHING1M_PATH, transform_dict["clothing1M"],mode='val')
		fashion_MNIST_test = fashionMNIST_dataset(FASHIONMNIST_PATH, transform_dict["fashion-MNIST"])

		train_dataset = [clothing1M_train]
		test_dataset = [fashion_MNIST_test]
		val_dataset = [clothing1M_val]
		train_len = len(clothing1M_train)
	
	elif args.dataset == 'CP':
		num_classes = 4
		transform_dict = {
			"C_Train": train_transform,
			"C_Task1": train_transform,
			"C_Task2": train_transform,
			"C_Task3": train_transform,
		}

		for d in args.validation_data:
			transform_dict[d] = val_transform

		C_Train = CP_dataset(CP_PATH, task_id=0, transform=transform_dict['C_Train'])
		C_Task1 = CP_dataset(CP_PATH, task_id=1, transform=transform_dict['C_Task1'])
		C_Task2 = CP_dataset(CP_PATH, task_id=2, transform=transform_dict['C_Task2'])
		C_Task3 = CP_dataset(CP_PATH, task_id=3, transform=transform_dict['C_Task3'])

		dataset_dict = {
			"C_Train": C_Train,
			"C_Task1": C_Task1,
			"C_Task2": C_Task2,
			"C_Task3": C_Task3,
		}

	if 'clothing' not in args.dataset:
		train_dataset, val_dataset, test_dataset, train_len = _construct_dataset_helper(
			args, dataset_dict, train_transform, val_transform
		)

	return train_dataset, test_dataset, val_dataset, num_classes, train_len

def add_noise_dataset_dict(args, d, dataset_d):
	if args.dataset == "terraincognita":
		num_classes = 10
		noise_lab_file = '%s_%s_noisy_%s_lab_%d.npz'%(args.noise_path+'terraInc/', d, args.noise_type, int(args.noise_ratio*100))
		if os.path.exists(noise_lab_file):
			noise_label = np.load(noise_lab_file)['label']
		else:
			noise_label = _get_labels_from_dataset(dataset_d)
			class_ind = {}
			for kk in range(10):
				class_ind[kk] = [i for i in range(len(noise_label)) if noise_label[i]==kk]
			if args.noise_type == 'random':
				noise_dict = {0:1, 2:5, 3:6, 4:7, 6:8, 9:5}
			else:
				noise_dict = {1:2, 2:4, 3:1, 7:9, 8:9, 9:7}
			for i in range(num_classes):
				if i not in noise_dict:
					continue
				indices = class_ind[i]
				np.random.shuffle(indices)
				for j, idx in enumerate(indices):
					if j < args.noise_ratio * len(indices):
						noise_label[idx] = noise_dict[i]
			print("Save noisy labels to %s ..."%noise_lab_file)        
			np.savez(noise_lab_file, label = noise_label)
		noisy_d = _reset_labels_to_dataset(dataset_d, noise_label)
		return noisy_d

def _get_labels_from_dataset(d):
	labels = []
	for i in range(len(d.imgs)):
		_, label_i = d.imgs[i]
		labels.append(label_i)
	return labels

def _reset_labels_to_dataset(d, labels):
	for i in range(len(d.imgs)):
		img_path, _ = d.imgs[i]
		d.imgs[i] = img_path, labels[i]
	return d

class clothing_dataset(Dataset):
	def __init__(
		self,
		root,
		transform,
		mode='train',
		num_class=5
	):

		self.root = root
		self.transform = transform
		self.train_labels = {}
		self.val_labels   = {}
		self.mode = mode
		# Select classes overlapping in clothing1M and fashion MNIST: [new index] fashion MNIST: clothing1M
		# [0]: 0 T-shirt/top : 0 T-shirt
		# [1]: 2 Pullover: 5 Hoodie
		# [2]: 3 Dress: 11 Dress
		# [3]: 4 Coat: 6 Windbreaker + 7 jacket +8 downcoat 
		# [4]: 6 Shirt: 1 Shirt
		# new_label_mapping: clothing1M old index: new index
		new_label_mapping = {0: 0, 5:1, 11:2, 6:3, 7:3, 8:3, 1:4}

		if self.mode == 'train':
			with open("%s/noisy_label_kv.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					entry = l.split()
					img_path = "%s/" % self.root + entry[0][7:]
					self.train_labels[img_path] = int(entry[1])
			
			train_imgs = []
			with open("%s/noisy_train_key_list.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					img_path = "%s/" % self.root + l[7:]
					train_imgs.append(img_path)
			random.shuffle(train_imgs)
			self.train_imgs = []
			self.train_new_labels = {}
			for impath in train_imgs:
				label = self.train_labels[impath]
				if label in new_label_mapping:
					self.train_imgs.append(impath)
					self.train_new_labels[impath] = new_label_mapping[label]
			random.shuffle(self.train_imgs)
			self.imgs = self.train_imgs
			self.label_dict = self.train_new_labels
		
		elif self.mode == 'clip_test': # test set for training on clip label source
			with open("%s/clean_label_kv.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					entry = l.split()
					img_path = "%s/" % self.root + entry[0][7:]
					self.val_labels[img_path] = int(entry[1])

			self.val_imgs = []
			with open("%s/clean_test_key_list.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					img_path = "%s/" % self.root + l[7:]
					self.val_imgs.append(img_path)
			self.imgs = self.val_imgs
			self.label_dict = self.val_labels
		
		else:
			with open("%s/clean_label_kv.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					entry = l.split()
					img_path = "%s/" % self.root + entry[0][7:]
					self.val_labels[img_path] = int(entry[1])

			self.val_imgs = []
			self.val_new_labels = {}
			with open("%s/clean_test_key_list.txt" % self.root, "r") as f:
				lines = f.read().splitlines()
				for l in lines:
					img_path = "%s/" % self.root + l[7:]
					label = self.val_labels[img_path]
					if label in new_label_mapping:
						self.val_imgs.append(img_path)
						self.val_new_labels[img_path] = new_label_mapping[label]
			self.imgs = self.val_imgs
			self.label_dict = self.val_new_labels

	def __getitem__(self, index):
		img_path = self.imgs[index]
		target = self.label_dict[img_path]
		image = Image.open(img_path).convert("RGB")
		img = self.transform(image)
		return img, target, index

	def __len__(self):
		return len(self.imgs)
	
class fashionMNIST_dataset(datasets.FashionMNIST):
	# Select classes overlapping in clothing1M and fashion MNIST: [new index] fashion MNIST: clothing1M
		# [0]: 0 T-shirt/top : 0 T-shirt
		# [1]: 2 Pullover: 5 Hoodie
		# [2]: 3 Dress: 11 Dress
		# [3]: 4 Coat: 6 Windbreaker + 7 jacket +8 downcoat 
		# [4]: 6 Shirt: 1 Shirt
		# new_label_mapping: fashionMNIST old index: new index
	def __init__(self, root, transform):
		super(fashionMNIST_dataset, self).__init__(root = root, transform=transform, download=True)
		new_label_mapping = {0:0, 2:1, 3:2, 4:3, 6:4}
		imgs, new_labels = [], []
		for i in range(len(self.data)):
			label = self.targets[i].item()
			if label in new_label_mapping:
				imgs.append(self.data[i])
				new_labels.append(new_label_mapping[label])
		self.data = imgs
		self.targets = new_labels

	def __getitem__(self, index):
		img, target = self.data[index], int(self.targets[index])
		img = Image.fromarray(img.numpy(), mode="L")
		img = img.convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target, index



class CP_dataset(Dataset):
	def __init__(self, root_dir, task_id, transform):
		self.root_dir = root_dir
		self.labels = {}
		self.classes_dict = {
				"BRD-A29260609": 0,
				"BRD-K04185004": 1,
				"BRD-K21680192": 2,
				"DMSO": 3,
		}
		self.transform = transform
		
		if task_id == 1:
			self.task_id = 'Task_one'
		elif task_id == 2:
			self.task_id = 'Task_two'
		elif task_id == 3:
			self.task_id = 'Task_three'
		else:
			self.task_id = 'Train'

		self.metadata = pd.read_csv(self.root_dir+'/cp.csv')
		self.data = []
		for i in range(len(self.metadata)):
			if self.metadata.loc[i, "train_test_split"] == self.task_id:
				self.data.append(self.metadata.loc[i, "file_path"])
				self.labels[self.metadata.loc[i, "file_path"]] = self.classes_dict[self.metadata.loc[i, "label"]]  
		
		self.samples = []
		for i in range(len(self.data)):
			self.samples.append((self.data[i], self.labels[self.data[i]]))
	
	def __getitem__(self, index):
		path, target = self.samples[index]
		path = CP_PATH+'/images/'+path
		sample = self.load_img(path)
		return sample, target, index

	
	def load_img(self, img_path):
		img = skimage.io.imread(img_path)
		img = np.reshape(img, (img.shape[0], 160, -1), order="F")
		img = transforms.ToTensor()(img)
		img = self.transform(img)
		return img

	def __len__(self):
		return len(self.samples)