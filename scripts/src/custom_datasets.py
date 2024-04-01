#!/usr/bin/env python

"""datasets.py : Contains pytorch dataset inherited classes for training pytorch NNs"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import torch
import numpy as np # linear algebra
import segmentation_models_pytorch as smp
import pandas as pd
import torch.nn as nn
from matplotlib import cm
import cv2

# class AL_Seg_dataset(Dataset):
#     def __init__(self, oracle_path, inp_df, init = True, transform=True, use_sam = True):
#         # Initialize the Dataset class
    
#         self.transform_flag = transform
#         self.img_size = (256,256) 
#         self.boxes = []
#         self.oracle_path = oracle_path
#         self.use_sam = use_sam
#         self.df = pd.DataFrame(columns=["images", "masks", "oracle"])
#         if init == True:
#             self.df["images"] = inp_df["images"].to_list()
#             self.df["masks"] = inp_df["masks"].to_list()
#             self.df["oracle"] = inp_df["oracle"].to_list()
            
#         # Ensure the number of images is equal to the number of labels
#         assert len(self.df["images"]) == len(self.df["masks"])

#     def __encode_image__(self, filepath):
#         # Helper function to encode the image
#         image_rgb = Image.open(filepath).convert("RGB")
#         # print("Chorus_dataset__encode_image__")
#         # image_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#         return image_rgb

#     def __len__(self):
#         # Return the length of the dataset
#         return len(self.labels)

#     def __transform__(self, image, mask):
#         # Apply transformations to the image and mask

#         # Resize the image and mask
#         # print("Chorus_dataset__transform__")
#         # image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC)
#         # mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_CUBIC)
        
#         resize = T.Resize(size=self.img_size, interpolation=Image.NEAREST)
#         image = resize(image)
#         mask = resize(mask)

#         # Transform the image to a tensor
#         image = TF.to_tensor(image)
#         # Transform the mask to a tensor
#         #Recomment this
#         # mask = TF.to_tensor(mask)

#         return image, mask

#     def __getitem__(self, index):
#         # Retrieve the label and image at the given index
#         # print("Chorus_dataset__getitem__")
#         label = None

#         if self.use_sam and os.path.isfile(self.df["oracle"][index]):
#             mask_path = self.df["oracle"][index]
                
#         else:
#             mask_path = self.labels[index]
        
#         if mask_path.endswith("npy"):
#             label = torch.Tensor(np.load(mask_path, allow_pickle=True))
#             # label = np.array(np.load(mask_path, allow_pickle=True), np.uint8)
#         else:
#             label = TF.to_tensor(cv2.imread(mask_path))
#             # label = cv2.imread(mask_path)
            
#         if self.images[index].endswith("npy"):
#             image = torch.Tensor(np.load(self.images[index], allow_pickle=True))
#             image = np.load(self.images[index], allow_pickle=True)
#         else:
#             image = self.__encode_image__(self.images[index])
            
#         identifier = self.images[index]

#         # boxes = torch.Tensor(np.load(self.boxes[index]))
#         # Reshape boxes

#         if self.transform_flag == True:
#             # Apply transformations if transform_flag is True
#             img, mask = self.__transform__(image, label)
#         else:
#             # Convert image to tensor without transformations
#             img = TF.to_tensor(image)
#             mask = label
#         mask = mask.clamp(max=1.0)

#         return {'image': img, 'mask': mask, 'id': identifier}

    
    

    
class Handler(Dataset):
    def __init__(self, X, Y, use_sam=True):
        """
        Custom dataset handler for image and mask data.

        Args:
            X (list): List of image file paths.
            Y (list): List of mask file paths.
            img_size (tuple): Size to resize the image and mask.
        """
        self.X = X
        self.Y = Y
        self.img_size = (256, 256) 

    def __transform__(self, image, mask):
        """
        Apply transformations to the image and mask.

        Args:
            image (PIL.Image): Image object.
            mask (ndarray): Mask data.

        Returns:
            tuple: Transformed image and mask.
        """
        # Resize the image and mask
        # print("Handler__transform__")
        # try:
        #     image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC)
        #     mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_CUBIC)
        # except:
        #     # print(mask.shape)
        #     pass
                
        resize = T.Resize(size=self.img_size, interpolation=Image.NEAREST)
        image = resize(image)
        mask = resize(mask)
        # if mask.shape == (1024, 1, 1024):
        #     mask = np.transpose(mask, (1, 2, 0))
            # print(153)        
        # Transform the image to a tensor
        image = TF.to_tensor(image)

        # Transform the mask to a tensor
        # mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        """
        Get the image, mask, and index at the given index.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Image, mask, and index.
        """
        # print("getting item with handler")
        # Retrieve the label and image at the given index
        # print("Handler__getitem__")
        label = None
        if self.Y[index].endswith("npy"):
            label = torch.Tensor(np.load(self.Y[index], allow_pickle=True))
            if label.dim() == 2:
                label = label.unsqueeze(0)
            # label = np.array(np.load(self.Y[index], allow_pickle=True), np.uint8)
        else:            
            # label = cv2.cvtColor(cv2.imread(self.Y[index]), cv2.COLOR_BGR2GRAY)
            # label = TF.to_tensor(cv2.cvtColor(cv2.imread(self.Y[index]), cv2.COLOR_BGR2GRAY))
            label = TF.to_tensor(Image.open(self.Y[index]).convert("L"))

        if self.X[index].endswith("npy"):
            # image = np.load(self.X[index], allow_pickle=True)
            try:
                image = Image.fromarray(np.load(self.X[index], allow_pickle=True)).convert("RGB")
            except:
                image = Image.fromarray(np.load(self.X[index], allow_pickle=True).astype(np.uint8)).convert("RGB")
            # print("npy shape :", image.size)
        else:
            # image = cv2.cvtColor(cv2.imread(self.X[index]), cv2.COLOR_BGR2RGB)
            image = Image.open(self.X[index]).convert("RGB")
            
        img, mask = self.__transform__(image, label)
        mask = mask.clamp(max=1.0)
        # if mask.shape == (1024, 1, 1024):
        #     mask = np.transpose(mask, (1, 2, 0))
            # print(191)
        return img, mask, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)

    
class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, df:pd.DataFrame=None, path:str = None, use_sam=True):
        """
        Data management class for handling labeled and unlabeled data.

        Args:
            X_train (list): List of training image file paths.
            Y_train (list): List of training mask file paths.
            X_test (list): List of test image file paths.
            Y_test (list): List of test mask file paths.
            handler (Handler): Instance of the Handler class for data transformation.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.use_sam = use_sam
        self.img_size = (256, 256)

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.df=df
        self.path=path

    def initialize_labels(self, num):
        """
        Initialize the labeled pool by randomly selecting a given number of samples.

        Args:
            num (int): Number of samples to initialize as labeled.

        Returns:
            None
        """
        # Generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        self.labeled_idxs[:num] = True

    def get_labeled_data(self):
        """
        Get the labeled data and their corresponding indices.

        Returns:
            tuple: Labeled indices and transformed labeled data.
        """
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        labeled_X = [self.X_train[idx] for idx in labeled_idxs]
        print(len(labeled_X))
        labeled_Y = []
        for idx in labeled_idxs:
            if self.use_sam and os.path.isfile(self.df["oracle"][idx]):
                labeled_Y.append(self.df["oracle"][idx])
            else:
                labeled_Y.append(self.Y_train[idx])
        # labeled_Y = [self.Y_train[idx] for idx in labeled_idxs]
        return labeled_idxs, self.handler(labeled_X, labeled_Y, self.use_sam)

    def get_unlabeled_data(self):
        """
        Get the unlabeled data and their corresponding indices.

        Returns:
            tuple: Unlabeled indices and transformed unlabeled data.
        """
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        unlabeled_X = [self.X_train[idx] for idx in unlabeled_idxs]
        unlabeled_Y = [self.Y_train[idx] for idx in unlabeled_idxs]
        return unlabeled_idxs, self.handler(unlabeled_X, unlabeled_Y, self.use_sam)

    def get_train_data(self):
        """
        Get the training data (labeled + unlabeled) and their corresponding indices.

        Returns:
            tuple: Labeled indices and transformed training data.
        """
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.use_sam)

    def get_test_data(self):
        """
        Get the test data and their corresponding indices.

        Returns:
            tuple: Transformed test data.
        """
        return self.handler(self.X_test, self.Y_test, self.use_sam)

    def cal_test_metrics(self, logits, mask):
        """
        Calculate evaluation metrics for the test data.

        Args:
            logits (torch.Tensor): Logits output of the model.
            mask (torch.Tensor): Ground truth mask.

        Returns:
            tuple: Intersection over Union (IoU) and F1-score.
        """
        prob_mask = logits.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Compute true positive, false positive, false negative, and true negative 'pixels' for each class
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        # Calculate IoU and F1-score
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        # loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # dice_loss = loss(logits, mask)

        return iou, accuracy, precision, recall, f1


