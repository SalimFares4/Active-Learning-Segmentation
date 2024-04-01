#!/usr/bin/env python

"""utils.py: module for helper functions"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"

import base64
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import torch
import numpy as np
from tqdm import tqdm
import numpy as np # linear algebra
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
import cv2
#from skimage.util.montage import montage2d as montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
from skimage.morphology import label
import supervision as sv
from segment_anything.utils.transforms import ResizeLongestSide




class Dataset(Dataset):
    def __init__(self, path, transform=True):
        # Initialize the Dataset class
        self.transform_flag = transform
        self.path = path
        self.images = []
        self.labels = []
        self.boxes = []
        
        # Iterate over the files in the 'masks' directory
        for file in os.listdir(os.path.join(self.path, 'masks')):
            fn = file.split('.npy')[0]
            
            # Append paths for images, labels, and boxes
            self.images.append(os.path.join(self.path, 'chorus', "{}.png".format(fn)))
            self.labels.append(os.path.join(self.path, 'masks', "{}.npy".format(fn)))
            #self.boxes.append(os.path.join(self.path, 'boxes', file))
        
        # Ensure the number of images is equal to the number of labels
        assert len(self.images) == len(self.labels)
        
    def __encode_image(self, filepath):
        # Helper function to encode the image
        img = Image.open(filepath).convert("RGB")
        return img
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.labels)
    
    def __transform__(self, image, mask):
        # Apply transformations to the image and mask
        
        # Resize the image and mask
        resize = T.Resize(size=(1024, 1024), interpolation=Image.NEAREST)
        image = resize(image)
        mask = resize(mask)
        
        # Transform the image to a tensor
        image = TF.to_tensor(image)
        # Transform the mask to a tensor
        # mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        # Retrieve the label and image at the given index
        label = torch.Tensor(np.load(self.labels[index]))
        image = self.__encode_image(self.images[index])
        
        #boxes = torch.Tensor(np.load(self.boxes[index]))
        # Reshape boxes 
        
        if self.transform_flag == True:
            # Apply transformations if transform_flag is True
            img, mask = self.__transform__(image, label)
        else:
            # Convert image to tensor without transformations
            img = TF.to_tensor(image)
            mask = label
        mask = mask.clamp(max = 1.0)
            
        return {'image': img, 'mask': mask}
    
    
def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+ encoded


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_boxes(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        
def plot(dataset, i):
    sample = dataset[i]
    image = sample['image']
    boxes = sample['boxes']
    mask = sample['mask']
    plt.figure(figsize=(10,10))
    plt.imshow(image.permute(1,2,0))
    show_boxes(boxes, plt.gca())
    show_mask(mask, plt.gca())
    plt.axis('off')
    plt.show()  
    
    
from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((1024, 1024), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def box_to_centroid(bbox):
    x, y, w, h = bbox  # Extract bounding box coordinates

    # Compute the centroid point
    centroid_x = x + (w // 2)
    centroid_y = y + (h // 2)
    centroid = (centroid_x, centroid_y)
    return centroid

def convert_image_to_uint(image):
    
    # Convert image to BGR format
    opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Normalize the image
    normalized_image = (opencv_image - np.min(opencv_image)) / (np.max(opencv_image) - np.min(opencv_image))

    # Convert the normalized image to uint8 format
    normalized_image_uint = (normalized_image * 255).astype(np.uint8)
    
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(normalized_image_uint, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def query_sam(img_cv, mask_predictor, device, boxes = None, points = None, labels = None):
    
    mask_predictor.set_image(img_cv)
    if boxes != None:
        boxes = torch.Tensor(boxes).to(device)

        transformed_boxes = mask_predictor.transform.apply_boxes_torch(boxes, img_cv.shape[:2])


        masks, scores, logits = mask_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        mask = masks.sum(axis = 0)
        return mask, scores
    
    else:
        masks, scores, logits = mask_predictor.predict(
            point_coords = points,
            point_labels = labels,
            mask_input=None,
            multimask_output=False
        )
        return masks, scores
    
    

    
    
def query_sam_decoder(bottleneck, boxes,  masks):
    #masks = np.array(boxes)
    #box_transformed = transform.apply_boxes(prompt_box,(1024, 1024))
    masks = torch.as_tensor(masks, dtype=torch.float, device=DEVICE)
    boxes = torch.as_tensor(boxes, dtype = torch.float, device = DEVICE)
    
    # Make sure 'sam' and 'prompt_encoder' are defined and accessible
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=masks,
        boxes=None,
        masks=None,
    )
   
    
    # Make sure 'mask_decoder' and 'postprocess_masks' are defined and accessible
    low_res_masks, scores = sam.mask_decoder(
        image_embeddings=bottleneck,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # Make sure 'sam' and 'postprocess_masks' are defined and accessible
    upscaled_masks = sam.postprocess_masks(low_res_masks, (1024, 1024), (1024, 1024))
    upscaled_masks = upscaled_masks.sum(axis=0)

    # Make sure 'normalize' and 'threshold' functions are defined and accessible
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    return upscaled_masks, binary_mask, scores


import math

def calculate_prediction_mask_entropy(prediction_mask):
    # Convert the prediction mask to integers
    prediction_mask = np.asarray(prediction_mask, dtype=np.int64)

    # Flatten the prediction mask array
    flattened_mask = np.ravel(prediction_mask)

    # Count the occurrence of each label in the prediction mask
    label_counts = np.bincount(flattened_mask)

    # Calculate the total number of labels
    total_labels = len(flattened_mask)

    # Calculate the probabilities
    probabilities = label_counts / total_labels

    # Filter out zero probabilities
    probabilities = probabilities[probabilities > 0]

    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def convert_box(bbox):
    box = np.array([
            bbox[1], 
            bbox[0], 
            bbox[3], 
            bbox[2]
        ])
       
    return box
