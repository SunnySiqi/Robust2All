from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy


class dg_clothing_dataset(Dataset):
    def __init__(
        self, sample_ratio,
        root,
        transform,
        mode,
        num_samples=0,
        pred=[],
        probability=[],
        paths=[],
        num_class=5,
    ):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.new_train_labels = {}
        self.test_labels  = {}
        self.val_labels   = {}
        self.val_new_labels = {}
        clothing1M_label_mapping = {0: 0, 5:1, 11:2, 6:3, 7:3, 8:3, 1:4}
        fashion_label_mapping = {0:0, 2:1, 3:2, 4:3, 6:4}

        with open("%s/noisy_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        
        for impath in self.train_labels:
            if self.train_labels[impath] in clothing1M_label_mapping:
                self.new_train_labels[impath] = clothing1M_label_mapping[self.train_labels[impath]]

        with open("%s/clean_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.val_labels[img_path] = int(entry[1])
        
        for impath in self.val_labels:
            if self.val_labels[impath] in clothing1M_label_mapping:
                self.val_new_labels[impath] = clothing1M_label_mapping[self.val_labels[impath]]
        
        save_file = 'pred_idx_clothing1M_aug.npz'
        if mode == 'all':
            train_imgs = []
            with open("%s/noisy_train_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if (
                    len(self.train_imgs) < num_samples
                    and label in clothing1M_label_mapping
                    and class_num[clothing1M_label_mapping[label]] < (num_samples / num_class)
                ):
                    self.train_imgs.append(impath)
                    class_num[clothing1M_label_mapping[label]] += 1
            random.shuffle(self.train_imgs)
        
        elif mode == 'labeled':
            train_imgs = paths
            pred_idx   = np.zeros(int(sample_ratio*num_samples))
            class_len  = int(sample_ratio*num_samples/num_class)
            size_pred  = 0
            class_ind  = {}

            ## Get the class indices
            for kk in range(num_class):
                class_ind[kk] = [i for i,x in enumerate(train_imgs) if self.new_train_labels[x]==kk]

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
            self.train_imgs  = [train_imgs[i] for i in pred_idx]
            probability[probability<0.5] = 0                        ## Weight Adjustment 
            self.probability = [1-probability[i] for i in pred_idx]
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx1 = np.load(save_file)['index']
            idx = list(range(num_samples))
            pred_idx = [x for x in idx if x not in pred_idx1] 
            self.train_imgs = [train_imgs[i] for i in pred_idx]                         
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif mode == "test":
            imgs, new_labels = [], []
            fashionMNIST = datasets.FashionMNIST(root = '../../Dataset/fashion-mnist/', transform=self.transform, download=True)
            for i in range(len(fashionMNIST.data)):
                label = fashionMNIST.targets[i].item()
                if label in fashion_label_mapping:
                    imgs.append(fashionMNIST.data[i])
                    new_labels.append((torch.tensor(fashion_label_mapping[label]).long()))
            self.test_imgs = imgs
            self.test_labels = new_labels
        
        elif mode == "val":
            self.val_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path =  "%s/" % self.root + l[7:]
                    label = self.val_labels[img_path]
                    if label in clothing1M_label_mapping:
                        self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.new_train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            
            return img1, img2, img3, img4, target, prob

        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4
            
        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.new_train_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, img_path

        elif self.mode == "test":
            img = self.test_imgs[index]
            target = self.test_labels[index]
            image = Image.fromarray(img.numpy(), mode="L").convert("RGB")
            img = self.transform(image)
            return img, target
            
        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.val_new_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class dg_clothing_dataloader:
    def __init__(
        self,
        root,
        batch_size,
        warmup_batch_size,
        num_batches,
        num_workers    ):
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root

        clothing1m_weak_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )


        clothing1m_strong_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )

        self.transforms = {
            "warmup": clothing1m_weak_transform,
            "unlabeled": [
                        clothing1m_strong_transform,
                        clothing1m_strong_transform,
                        clothing1m_weak_transform,
                        clothing1m_weak_transform
                    ],
            "labeled": [
                        clothing1m_strong_transform,
                        clothing1m_strong_transform,
                        clothing1m_weak_transform,
                        clothing1m_weak_transform
                    ]
        }
        self.transforms_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )

        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def run(self, sample_r, mode,  pred=[], prob=[], paths=[]):
        if mode == "warmup":
            warmup_dataset = dg_clothing_dataset(sample_r,
                self.root,
                transform=self.transforms["warmup"],
                mode="all",
                num_samples=self.num_batches * self.warmup_batch_size,
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = dg_clothing_dataset( sample_r,
                self.root, 
                transform=self.transforms["labeled"],
                mode="labeled", 
                num_samples=self.num_batches * self.batch_size,
                pred=pred,
                probability=prob,
                paths=paths
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True
            )
            unlabeled_dataset = dg_clothing_dataset(sample_r,
                self.root, 
                transform = self.transforms["unlabeled"],
                mode = "unlabeled",
                num_samples = self.num_batches * self.batch_size,
                pred = pred,
                probability=prob,
                paths=paths
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=self.num_workers, drop_last = True
            )
            return labeled_loader, unlabeled_loader

        elif mode == "eval_train":
            eval_dataset = dg_clothing_dataset( sample_r,
                self.root, 
                transform=self.transforms_test,
                mode="all",
                num_samples=self.num_batches * self.batch_size,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*4,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = dg_clothing_dataset(
                sample_r,self.root,  transform=self.transforms_test, mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "val":
            val_dataset = dg_clothing_dataset(
                sample_r, self.root ,transform=self.transforms_val, mode="val"
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader
