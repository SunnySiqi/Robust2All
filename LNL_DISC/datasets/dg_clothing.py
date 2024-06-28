# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : Clothing1M.py

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
from PIL import Image
import torch
from .randaugment import TransformFixMatchLarge
import os


class dg_clothing_dataset(Dataset):

    def __init__(self, root_dir, mode, num_samples=0):
        # Select classes overlapping in clothing1M and fashion MNIST: [new index] fashion MNIST: clothing1M
        # [0]: 0 T-shirt/top : 0 T-shirt
        # [1]: 2 Pullover: 5 Hoodie
        # [2]: 3 Dress: 11 Dress
        # [3]: 4 Coat: 6 Windbreaker + 7 jacket +8 downcoat 
        # [4]: 6 Shirt: 1 Shirt
        # new_label_mapping: clothing1M old index: new index
        clothing1M_label_mapping = {0: 0, 5:1, 11:2, 6:3, 7:3, 8:3, 1:4}
        fashion_label_mapping = {0:0, 2:1, 3:2, 4:3, 6:4}


        self.root_dir = root_dir
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.num_samples = num_samples
        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # import ipdb; ipdb.set_trace()
        self.transform_fixmatch = TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
        with open('%s/noisy_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0][7:]
                if os.path.exists(self.root_dir + '/' + img_path):
                    self.train_labels[img_path] = int(entry[1])

        with open('%s/clean_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0][7:]
                if os.path.exists(self.root_dir + '/' + img_path):
                    self.val_labels[img_path] = int(entry[1])

        if mode == 'train' or mode == 'all':
            self.train_imgs = []
            self.train_new_labels = {}
            self.labels = []
            #self.labels = torch.zeros(len(self.train_labels)).long()
            with open('%s/noisy_train_key_list.txt' % self.root_dir,'r') as f:
                lines = f.read().splitlines()
                for i, l in enumerate(lines):
                    img_path = l[7:]
                    label = self.train_labels[img_path]
                    if label in clothing1M_label_mapping:
                        self.train_imgs.append(img_path)
                        self.train_new_labels[img_path] = clothing1M_label_mapping[label]
                        self.labels.append(torch.tensor(clothing1M_label_mapping[label]).long())
            self.index_dict = {}
            for i in range(len(self.train_imgs)):
                self.index_dict[self.train_imgs[i]] = i

        elif mode == 'test' or mode == 'test_ind':
            imgs, new_labels = [], []
            fashionMNIST = datasets.FashionMNIST(root = '../../Dataset/fashion-mnist/', transform=self.transform_test, download=True)
            for i in range(len(fashionMNIST.data)):
                label = fashionMNIST.targets[i].item()
                if label in fashion_label_mapping:
                    imgs.append(fashionMNIST.data[i])
                    new_labels.append((torch.tensor(fashion_label_mapping[label]).long()))
            self.test_imgs = imgs
            self.test_labels = new_labels

        elif mode == 'val':
            self.val_imgs = []
            self.val_new_labels = {}
            self.labels = []
            with open('%s/clean_test_key_list.txt' % self.root_dir,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = l[7:]
                    label = self.val_labels[img_path]
                    if label in clothing1M_label_mapping:
                        self.val_imgs.append(img_path)
                        self.val_new_labels[img_path] = clothing1M_label_mapping[label]
                        self.labels.append(torch.tensor(clothing1M_label_mapping[label]).long())
    

    def sample_subset(self, num_class=5):  #sample a class-balanced subset
        random.shuffle(self.train_imgs)
        class_num = torch.zeros(num_class)
        self.train_imgs_subset = []
        for impath in self.train_imgs:
            label = self.train_new_labels[impath]
            if class_num[label] < (self.num_samples / num_class) and len(
                    self.train_imgs_subset) < self.num_samples:
                self.train_imgs_subset.append(impath)
                class_num[label] += 1
        random.shuffle(self.train_imgs_subset)
        return

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs_subset[index]
            idx = self.index_dict[img_path]
            target = self.train_new_labels[img_path]
            img_path = self.root_dir + '/' + img_path
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, idx
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            idx = self.index_dict[img_path]
            target = self.train_new_labels[img_path]
            img_path = self.root_dir + '/' + img_path
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, idx
        elif self.mode == 'test':
            img = self.test_imgs[index]
            target = self.test_labels[index]
            image = Image.fromarray(img.numpy(), mode="L").convert("RGB")
            img = self.transform_test(image)
            return img, target
        elif self.mode == 'test_ind':
            img = self.test_imgs[index]
            target = self.test_labels[index]
            image = Image.fromarray(img.numpy(), mode="L").convert("RGB")
            img = self.transform_test(image)
            return img, target, index
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.val_new_labels[img_path]
            img_path = self.root_dir + '/' + self.val_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_val(image)
            return img, target

    def __len__(self):
        if self.mode == 'test' or self.mode == 'test_ind':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)
        elif self.mode == 'train':
            return len(self.train_imgs_subset)
        elif self.mode == 'all':
            return len(self.train_imgs)


class dg_clothing_dataloader():
    def __init__(self, root_dir, batch_size, num_workers, num_batches=1000):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.num_batches = num_batches
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])

    def run_all(self):
        train_dataset = dg_clothing_dataset(root_dir=self.root_dir,
                                         mode='all',
                                         num_samples=self.num_batches *
                                         self.batch_size)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = dg_clothing_dataset(root_dir=self.root_dir,
                                        mode='test',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, test_loader

    def run(self):
        train_dataset = dg_clothing_dataset(root_dir=self.root_dir,
                                         mode='train',
                                         num_samples=self.num_batches *
                                         self.batch_size)
        train_dataset.sample_subset()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = dg_clothing_dataset(root_dir=self.root_dir,
                                        mode='test',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        eval_dataset = dg_clothing_dataset(root_dir=self.root_dir,
                                        mode='val',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        eval_loader = DataLoader(dataset=eval_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, eval_loader, test_loader

