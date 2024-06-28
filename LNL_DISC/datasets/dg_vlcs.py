from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
from PIL import Image, ImageFile
import torch
from .randaugment import TransformFixMatchLarge
import os
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

VLCS_dict = {'bird':0, 'car':1, 'chair':2, 'dog':3, 'person':4}
class dg_vlcs_dataset(Dataset):

    def __init__(self, root_dir, mode, test_domain, train_domain):

        self.root = root_dir
        self.mode = mode
        self.test_domain = test_domain
        self.train_domain = train_domain
        self.train_labels = {}
        self.test_labels = {}
        self.domain_labels = {}
        self.transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )
        # import ipdb; ipdb.set_trace()
        self.transform_fixmatch = TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))

        if mode == 'test' or mode == 'test_ind':
            self.test_labels, _ = self.read_data_single_domain(self.test_domain[0])
            self.test_imgs = list(self.test_labels)
        
        else:
            for d in self.train_domain:
                labels, d_labels = self.read_data_single_domain(d)
                self.train_labels.update(labels)
                self.domain_labels.update(d_labels)
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

    
    def read_data_single_domain(self, domain):
        labels = {}
        domain_labels = {}
        data_dir = os.path.join(self.root, domain)
        classes = os.listdir(data_dir)
        for c in classes:
            label_idx = VLCS_dict[c]
            cls_data_dir = os.path.join(data_dir, c)
            cls_data_files = os.listdir(cls_data_dir)
            for f in cls_data_files:
                labels[os.path.join(cls_data_dir, f)] = label_idx
                domain_labels[os.path.join(cls_data_dir, f)] = domain
        return labels, domain_labels

    def sample_subset(self, num_class=5):  #sample a class-balanced subset
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

    def __getitem__(self, index):
        if self.mode == 'train':
            #img_path = self.train_imgs_subset[index]
            img_path = self.train_imgs_split[index]
            idx = self.index_dict[img_path]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, idx
        elif self.mode == 'all':
            img_path = self.train_imgs_split[index]
            idx = self.index_dict[img_path]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, idx
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target
        elif self.mode == 'test_ind':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target, index
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_val(image)
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


class dg_vlcs_dataloader():
    def __init__(self, root_dir, batch_size, num_workers, test_domain, train_domain):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.test_domain = test_domain
        self.train_domain = train_domain

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
        train_dataset = dg_vlcs_dataset(root_dir=self.root_dir,
                                         mode='all',
                                         test_domain=self.test_domain,
                                         train_domain=self.train_domain)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = dg_vlcs_dataset(root_dir=self.root_dir,
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
        train_dataset = dg_vlcs_dataset(root_dir=self.root_dir,
                                         mode='train',
                                         test_domain=self.test_domain,
                                         train_domain=self.train_domain)
        #train_dataset.sample_subset()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = dg_vlcs_dataset(root_dir=self.root_dir,
                                        mode='test',
                                        test_domain=self.test_domain,
                                        train_domain=self.train_domain)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        eval_dataset = dg_vlcs_dataset(root_dir=self.root_dir,
                                        mode='val',
                                        test_domain=self.test_domain,
                                        train_domain=self.train_domain)
        eval_loader = DataLoader(dataset=eval_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, eval_loader, test_loader

