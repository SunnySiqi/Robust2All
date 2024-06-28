# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import torchvision.datasets as datasets
import random
import torchvision.transforms as transforms
import pandas as pd
import skimage.io
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    # Noisy source
    "Clothing",
    "CP"
]
# CLOTHING1M_PATH
# FashionMNIST_PATH
# CP_PATH

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 1  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,)),
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ["0", "1", "2"]


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ["0", "1", "2"]


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        """
        Args:
            root: root dir for saving MNIST dataset
            environments: env properties for each dataset
            dataset_transform: dataset generator function
        """
        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(environments)):
            images = original_images[i :: len(environments)]
            labels = original_labels[i :: len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["+90%", "+80%", "-90%"]

    def __init__(self, root):
        super(ColoredMNIST, self).__init__(
            root,
            [0.1, 0.2, 0.9],
            self.color_dataset,
            (2, 28, 28),
            2,
        )

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["0", "15", "30", "45", "60", "75"]

    def __init__(self, root):
        super(RotatedMNIST, self).__init__(
            root,
            [0, 15, 30, 45, 60, 75],
            self.rotate_dataset,
            (1, 28, 28),
            10,
        )

    def rotate_dataset(self, images, labels, angle):
        rotation = T.Compose(
            [
                T.ToPILImage(),
                T.Lambda(lambda x: rotate(x, angle, fill=(0,), resample=Image.BICUBIC)),
                T.ToTensor(),
            ]
        )

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for environment in environments:
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir)

# class noisy_PACS(PACS):
#     NOISE_RATIO = 0.1

#     def __getitem__(self, index):
#         """
#         Return: sub-dataset for specific domain
#         """
#         print(self.datasets[index])
#         (x, clean_y) = self.datasets[index]
#         y = clean_y
#         np.random.seed(47)
#         if np.random.uniform() < NOISE_RATIO:
#             while y == clean_y:
#                 y = np.random.randint(0, self.num_classes)
#         return x,y


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    N_STEPS = 15001
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir)


class Clothing(MultipleDomainDataset):
    CHECKPOINT_FREQ = 200
    #ENVIRONMENTS = ['fashion-MNIST','clothing1M_noisy', 'clothing1M_clean']
    ENVIRONMENTS = ['fashion-MNIST','clothing1M_noisy']
    
    def __init__(self, root):
        super().__init__()
        self.environments = ['fashion-MNIST', 'clothing1M_noisy']
        clothing1M_noisy = clothing_dataset(CLOTHING1M_PATH, noisy=True)
        fashion_MNIST = fashionMNIST_dataset(FashionMNIST_PATH)
        self.datasets = [fashion_MNIST, clothing1M_noisy]
        self.input_shape = (3, 224, 224)
        self.num_classes = 5


class CP(MultipleDomainDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['C_Train', 'C_Task1', 'C_Task2', 'C_Task3']
    
    def __init__(self, root):
        super().__init__()
        self.environments = ['C_Train', 'C_Task1', 'C_Task2', 'C_Task3']
        C_Train = CP_dataset(CP_PATH, task_id=0)
        C_Task1 = CP_dataset(CP_PATH, task_id=1)
        C_Task2 = CP_dataset(CP_PATH, task_id=2)
        C_Task3 = CP_dataset(CP_PATH, task_id=3)
        self.datasets = [C_Train, C_Task1, C_Task2, C_Task3]
        self.input_shape = (5, 224, 224)
        self.num_classes = 4
    
class clothing_dataset(Dataset):
    def __init__(
        self,
        root,
        noisy=False,
        complete=False
    ):

        self.root = root
        self.transform = None
        self.labels = {}
        # Select classes overlapping in clothing1M and fashion MNIST: [new index] fashion MNIST: clothing1M
        # [0]: 0 T-shirt/top : 0 T-shirt
        # [1]: 2 Pullover: 5 Hoodie
        # [2]: 3 Dress: 11 Dress
        # [3]: 4 Coat: 6 Windbreaker + 7 jacket +8 downcoat 
        # [4]: 6 Shirt: 1 Shirt
        # new_label_mapping: clothing1M old index: new index
        new_label_mapping = {0: 0, 5:1, 11:2, 6:3, 7:3, 8:3, 1:4}

        if noisy:
            self.train_labels = {}
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
            if complete:
                self.imgs = train_imgs
                self.label_dict = self.train_labels
            else:
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
        else:
            self.val_labels = {}
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
                    if not complete and label in new_label_mapping:
                        self.val_imgs.append(img_path)
                        self.val_new_labels[img_path] = new_label_mapping[label]
                    elif complete:
                        self.val_imgs.append(img_path)
                        self.val_new_labels[img_path] = label
            self.imgs = self.val_imgs
            self.label_dict = self.val_new_labels

        
        self.samples = []
        for i in range(len(self.imgs)):
            self.samples.append((self.imgs[i], self.label_dict[self.imgs[i]]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
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
    def __init__(self, root, transform=None):
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

        self.samples = []
        for i in range(len(self.data)):
            self.samples.append((self.data[i], self.targets[i]))
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.fromarray(path.numpy(), mode="L")
        sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class CP_dataset(Dataset):
	def __init__(self, root_dir, task_id):
		self.root_dir = root_dir
		self.labels = {}

		self.classes_dict = {
				"BRD-A29260609": 0,
				"BRD-K04185004": 1,
				"BRD-K21680192": 2,
				"DMSO": 3,
		}
        
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
		return sample, target

	
	def load_img(self, img_path):
		img = skimage.io.imread(img_path)
		img = np.reshape(img, (img.shape[0], 160, -1), order="F")
		img = transforms.ToTensor()(img)
		return img

	def __len__(self):
		return len(self.samples)
