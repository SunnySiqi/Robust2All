from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
from PIL import Image, ImageFile
import json
import torch
import os
import pickle
from autoaugment import CIFAR10Policy, ImageNetPolicy
ImageFile.LOAD_TRUNCATED_IMAGES = True

VLCS_dict = {'bird':0, 'car':1, 'chair':2, 'dog':3, 'person':4}
class dg_VLCS_dataset(Dataset):
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
        num_class=5
    ):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.test_domain = test_domain
        self.train_domain = train_domain
        self.train_labels = {}
        self.test_labels  = {}
        self.domain_labels = {}

        if mode == 'test':
            self.test_labels, _ = self.read_data_single_domain(self.test_domain[0])
            self.test_imgs = list(self.test_labels)
        else:
            for d in self.train_domain:
                labels, d_labels = self.read_data_single_domain(d)
                self.train_labels.update(labels)
                self.domain_labels.update(d_labels)
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

                elif mode == 'labeled':
                    ori_class_domain_dict = {}
                    class_inds = {}
                    prob_class_domain_dict = {}
                    # for img_path in self.train_imgs_split:
                    #     label = self.train_labels[img_path]
                    #     domain = self.domain_labels[img_path]
                    #     if label in ori_class_domain_dict:
                    #         if domain in ori_class_domain_dict[label]:
                    #             ori_class_domain_dict[label][domain] += 1
                    #         else:
                    #             ori_class_domain_dict[label][domain] = 1
                    #     else:
                    #         ori_class_domain_dict[label] = {}
                    #         ori_class_domain_dict[label][domain] = 1
                    # print("Original CLASS_DOMAIN_DICT!!!!!", ori_class_domain_dict)
                    for kk in range(num_class):
                        class_inds[kk] = {}
                        prob_class_domain_dict[kk] = {}
                        for d in list(set(self.domain_labels.values())):
                            class_inds[kk][d] = [i for i,x in enumerate(self.train_imgs_split) if self.train_labels[x]==kk and self.domain_labels[x] == d]
                            prob_class_domain_dict[kk][d] = probability[class_inds[kk][d]]
                    # with open('prob_dist_'+save_file[:-4]+'.pkl', 'wb') as f:
                    #     pickle.dump(prob_class_domain_dict, f)


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


                    # class_domain_dict = {}
                    # for idx in pred_idx:
                    #     img_path = train_imgs[idx]
                    #     label = self.train_labels[img_path]
                    #     domain = self.domain_labels[img_path]
                    #     if label in class_domain_dict:
                    #         if domain in class_domain_dict[label]:
                    #             class_domain_dict[label][domain] += 1
                    #         else:
                    #             class_domain_dict[label][domain] = 1
                    #     else:
                    #         class_domain_dict[label] = {}
                    #         class_domain_dict[label][domain] = 1
                    # print("CLASS_DOMAIN_DICT after selection!!!!!", class_domain_dict)


                elif self.mode == "unlabeled":
                    train_imgs = self.train_imgs_split
                    num_samples = len(train_imgs)
                    pred_idx1 = np.load(save_file)['index']
                    idx = list(range(num_samples))
                    pred_idx = [x for x in idx if x not in pred_idx1] 
                    self.mode_train_imgs = [train_imgs[i] for i in pred_idx]                         
                    print("%s data has a size of %d"%(self.mode,len(self.mode_train_imgs)))

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

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.mode_train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            
            return img1, img2, img3, img4, target, prob

        elif self.mode == "unlabeled":
            img_path = self.mode_train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4
            
        elif self.mode == "all":
            img_path = self.mode_train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, img_path

        elif self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target
            
        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            return img1, img2, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.mode_train_imgs)


class dg_VLCS_dataloader:
    def __init__(
        self,
        root,
        batch_size,
        warmup_batch_size,
        num_workers,
        test_domain,
        train_domain,
        save_file):

        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root = root
        self.train_domain = train_domain
        self.test_domain = test_domain
        self.save_file = save_file

        vlcs_weak_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ]
        )


        vlcs_strong_transform = transforms.Compose(
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
            "warmup": vlcs_weak_transform,
            "unlabeled": [
                        vlcs_strong_transform,
                        vlcs_strong_transform,
                        vlcs_weak_transform,
                        vlcs_weak_transform
                    ],
            "labeled": [
                        vlcs_strong_transform,
                        vlcs_strong_transform,
                        vlcs_weak_transform,
                        vlcs_weak_transform
                    ]
        }
        # self.transforms_val = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
        #         ),
        #     ]
        # )

        # self.transforms_test = self.transforms_val

        transforms_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )

        self.transforms_val = [transforms_val, vlcs_strong_transform]

        self.transforms_test = transforms_val

    def run(self, sample_r, mode,  pred=[], prob=[], paths=[]):
        if mode == "warmup":
            warmup_dataset = dg_VLCS_dataset(sample_r,
                self.root,
                transform=self.transforms["warmup"],
                mode="all",
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = dg_VLCS_dataset( sample_r,
                self.root, 
                transform=self.transforms["labeled"],
                mode="labeled", 
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file,
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
            unlabeled_dataset = dg_VLCS_dataset(sample_r,
                self.root, 
                transform = self.transforms["unlabeled"],
                mode = "unlabeled",
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file,
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
            eval_dataset = dg_VLCS_dataset( sample_r,
                self.root, 
                transform=self.transforms_test,
                mode="all",
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*4,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        elif mode == "test":
            test_dataset = dg_VLCS_dataset(
                sample_r,self.root,  transform=self.transforms_test, mode="test",
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "val":
            val_dataset = dg_VLCS_dataset(
                sample_r, self.root ,transform=self.transforms_val, mode="val",
                test_domain=self.test_domain,
                train_domain=self.train_domain,
                save_file=self.save_file
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader
