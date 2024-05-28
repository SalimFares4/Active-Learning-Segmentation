#!/usr/bin/env python

"""strategies.py contains strategies for Active Learning
Adapted from https://github.com/ej0cl6/deep-active-learning to fit segmentation task in the use case """

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


import numpy as np
import torch
import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
import tqdm
# from sklearn.neighbors import NearestNeighbors
###########
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import cv2
import supervision as sv
import segmentation_models_pytorch as smp
from models import Net



class SAMOracle():
    
    def __init__(self,
                 device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                 model_type = "vit_h",
                 checkpoint_path = os.path.join("../sam","sam_vit_h_4b8939.pth"),
                 model = None,
                 img_size=(256, 256)
                ):
        
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        self.mask_predictor = SamPredictor(self.model)
        
        self.mask_generator = SamAutomaticMaskGenerator(model=self.model,
                                                        # points_per_side=32,
                                                        pred_iou_thresh=0.75,
                                                        stability_score_thresh=0.70,
                                                        # crop_n_layers=1,
                                                        # crop_n_points_downscale_factor=2,
                                                        # min_mask_region_area=100,  # Requires open-cv to run post-processing
                                                    )
        self.img_size=img_size        
        self.default_box = {"x": 0, "y": 0,"width": img_size[0], "height": img_size[1], "label": ''}
        
        
    
    def get_mask(self, img_path = None, img_rgb=None, boxes=[], multimask_output=False):
        if len(boxes) == 0:
            return np.zeros((1, self.img_size[0], self.img_size[1]), dtype=np.uint8)
        else:
            if img_rgb is None:
                try:
                    if img_path.endswith("npy"):
                        img_rgb = np.load(img_path, allow_pickle=True)
                    else:
                        image_bgr = cv2.imread(img_path)
                        resized = cv2.resize(image_bgr, self.img_size, interpolation=cv2.INTER_CUBIC)
                        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                except:
                    print(img_path)
                    image_bgr = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
            self.mask_predictor.set_image(img_rgb)
            
            if len(boxes) <1:
                boxes = []
                boxes.append(np.array([
                            self.default_box['x'],
                            self.default_box['y'],
                            self.default_box['x'] + self.default_box['width'],
                            self.default_box['y'] + self.default_box['height']]))
                boxes = np.array(boxes)
            boxes = torch.Tensor(boxes).to(self.device)
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(boxes, img_rgb.shape[:2])

            masks, scores, logits = self.mask_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes=transformed_boxes,
                multimask_output=multimask_output   
            )
            mask = masks.sum(axis = 0).cpu().numpy()
            mask = np.array(mask>0, dtype=np.uint8)
            return mask
    
    def get_multimask(self, img_path = None, img_rgb=None, boxes=[], multimask_output=True):
        
        if img_rgb is None:
            try:
                image_bgr = cv2.imread(img_path)
                resized = cv2.resize(image_bgr, self.img_size, interpolation=cv2.INTER_CUBIC)
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            except:
                print(img_path)
                image_bgr = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
        self.mask_predictor.set_image(img_rgb)
        
        if len(boxes) <1:
            boxes = []
            boxes.append(np.array([
                        self.default_box['x'],
                        self.default_box['y'],
                        self.default_box['x'] + self.default_box['width'],
                        self.default_box['y'] + self.default_box['height']]))
            boxes = np.array(boxes)
        boxes = torch.Tensor(boxes).to(self.device)
        transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(boxes, img_rgb.shape[:2])

        masks, scores, logits = self.mask_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes=transformed_boxes,
            multimask_output=multimask_output   
        )
        return masks
    
    def get_boxes(self, mask):
        if torch.is_tensor(mask):
            mask = mask.numpy()
            mask = np.array(mask, np.uint8)
        # _, thresh = cv2.threshold(mask, 0.5, 1, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            box = np.array([x, y, x+w, y+h])
            cnts.append(box)
        return np.array(cnts)    
    
    def generateMasks(self, img_path):
        img = cv2.imread(img_path)
        generated_masks = self.mask_generator.generate(img)
        list_generated_masks=[]
        for mask in generated_masks:
            list_generated_masks.append(mask["segmentation"])
        return list_generated_masks
    
    def annotate_mask(self, img_path:str, boxes=[]):
        if len(boxes) == 0:
            boxes.append(np.array([
                        self.default_box['x'],
                        self.default_box['y'],
                        self.default_box['x'] + self.default_box['width'],
                        self.default_box['y'] + self.default_box['height']]))
        boxes = torch.Tensor(np.array(boxes)).to(self.device)

        image_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
        mask = self.get_mask(img_rgb=img_rgb, boxes=boxes)
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask),
                                   mask=mask)
        
        detections = detections[detections.area == np.max(detections.area)]

        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        return segmented_image


class Strategy:
    def __init__(self, dataset, net, sam:SAMOracle, params):
    # def __init__(self, dataset, net):        
        """
        Initializes the Strategy class.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        self.dataset = dataset
        self.net = net
        self.sam = sam
        self.human_envolved = 0
        self.sam_failed = []
        self.params=params

    def query(self, n):
        """
        Selects a subset of unlabeled samples to query for labeling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        pass

    def update(self, pos_idxs, start_sam=False, use_predictor=False, use_generator=False, neg_idxs=None, round=1):
        """
        Updates the labeled indices in the dataset.

        Args:
            pos_idxs (ndarray): The indices of positively labeled samples.
            neg_idxs (ndarray or None): The indices of negatively labeled samples. Defaults to None.
        """
        self.dataset.labeled_idxs[pos_idxs] = True
        
        # for each to_be_added idx 
        # Check if we've already added the path 
        # If not, initialize path
        # Check if oracle(sam) has already annotated it.
        # If not, get the oracle_mask of the image using sam
        #save the oracle_mask in processed/oracle folder.
        #add the path of the mask to df["oracle_apth"]
        if start_sam:
            self.sam_failed = []
            for idx in pos_idxs:
                if not os.path.isfile(self.dataset.df["oracle"][idx]):
                    # path = self.dataset.df["oracle"][idx].split("/")
                    # path[-2] = f'oracle_sam_gen_{self.params["img_size"][0]}_{round}'
                    # parent_dir = "/".join(path[:-1])
                    # if not os.path.exists(parent_dir):
                    #     os.makedirs(parent_dir)
                    # path = "/".join(path)
                    if not use_predictor:
                        sam_generated_masks=self.sam.generateMasks(self.dataset.df["images"][idx])
                        majority =  self.generatorSelection(sam_generated_masks)
                        # np.save(path, majority.squeeze())
                        np.save(self.dataset.df["oracle"][idx], majority.squeeze())
        
                    elif not use_generator:
                        model_predicted_mask = self.predict(self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"]))[0]
                        model_predicted_mask = (model_predicted_mask.squeeze().cpu().sigmoid()> 0.5).float()                                                    
                        boxes = self.sam.get_boxes(model_predicted_mask)
        
                        if len(boxes)>200:
                            boxes = boxes[:200]
            
                        sam_predicted_mask = self.sam.get_mask(img_path=self.dataset.df["images"][idx], boxes=boxes)
                        # print(sam_predicted_mask.shape)        
                        # np.save(path,sam_predicted_mask.squeeze())
                        np.save(self.dataset.df["oracle"][idx],sam_predicted_mask.squeeze())
        
                    else:
                        model_predicted_mask = self.predict(self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"]))[0]
                        model_predicted_mask = (model_predicted_mask.squeeze().cpu().sigmoid()> 0.5).float()                            
                        sam_generated_masks=self.sam.generateMasks(self.dataset.df["images"][idx])
                        boxes = self.sam.get_boxes(model_predicted_mask)
        
                        if len(boxes)>200:
                            boxes = boxes[:200]
            
                        sam_predicted_mask = self.sam.get_mask(img_path=self.dataset.df["images"][idx], boxes=boxes)
            
                        sam_predicted_mask = sam_predicted_mask.squeeze()
                        sam_generated_masks.append(sam_predicted_mask)
                        majority = self.generatorSelection(sam_generated_masks)
                        
                        # np.save(path, majority.squeeze())
                        np.save(self.dataset.df["oracle"][idx], majority.squeeze())
                        
        # if neg_idx return the path in 'oracle' to "empty"
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False
            
    def update_voting(self, pos_idxs, start_sam=False, use_predictor=False, use_generator=False, neg_idxs=None, round=0):
        """
        Updates the labeled indices in the dataset.

        Args:
            pos_idxs (ndarray): The indices of positively labeled samples.
            neg_idxs (ndarray or None): The indices of negatively labeled samples. Defaults to None.
        """
        self.dataset.labeled_idxs[pos_idxs] = True
        states_path = "/root/Master_Thesis/scripts/notebooks/"
        if start_sam:
            self.sam_failed = []
            for idx in pos_idxs:
                model_predicted_masks = []
                sam_predicted_masks = []
                if not os.path.isfile(self.dataset.df["oracle"][idx]):                                
   
                    handler = self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"])
      
                    model_predicted_masks = self.getVotes(9, self.params, handler, round, pos_idxs)
                    current_mask = self.predict(self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"]))[0]
                    model_predicted_masks.append((current_mask.squeeze().cpu().sigmoid()> 0.5).float())
                    
                    for mask in model_predicted_masks:                            
                        boxes = self.sam.get_boxes(mask)
    
                        if len(boxes)>200:
                            boxes = boxes[:200]
        
                        sam_predicted_masks.append(self.sam.get_mask(img_path=self.dataset.df["images"][idx], boxes=boxes))
                    
                    np_sam = np.array(sam_predicted_masks).sum(axis=0)
                    
                    
                    majority = np.array((np_sam.squeeze() > 5), dtype=np.float32)
                    # path = self.dataset.df["oracle"][idx].split("/")
                    # path[-2] = f'oracle_mv_{self.params["img_size"][0]}_{round}'
                    # parent_dir = "/".join(path[:-1])
                    # if not os.path.exists(parent_dir):
                    #     os.makedirs(parent_dir)
                    # path = "/".join(path)
                    
                    # # np.save(path, majority.squeeze())
                    np.save(self.dataset.df["oracle"][idx], majority.squeeze())
    
        else: 
            self.human_envolved = len(pos_idxs)
        # if neg_idx return the path in 'oracle' to "empty"
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False
        
        return []
    
    def getVotes(self, models_num, params, handler, round, pos_idxs):
        masks = []
        model = smp.create_model(
            'Unet', encoder_name='resnet34', in_channels=3, classes = 1
            )
        net = Net(model, params, device = torch.device("cuda"))
        for i in range(1, models_num+1):
            dir = f'{params["model_path"]}_{round}'
            fname = f'{dir}/model_{i}.pt'
            
            if not os.path.isfile(fname):
                self.dataset.labeled_idxs[pos_idxs] = False
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                init_path = f'{params["model_path"]}_0/model_{i}.pt'
                if os.path.isfile(init_path):
                    net.net.load_state_dict(torch.load(init_path))
                else:
                    torch.save(net.net.state_dict(), init_path)
                print(f"Training model_{i} for voting")
                net.train(labeled_data)
                
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    
                torch.save(net.net.state_dict(),fname)
                self.dataset.labeled_idxs[pos_idxs] = True
            net.net.load_state_dict(torch.load(fname))
            logits = net.predict(handler)[0]
            mask = (logits.squeeze().cpu().sigmoid()> 0.5).float()
            masks.append(mask)
        return masks
    
    def update_weighted_voting(self, pos_idxs, start_sam=False, use_predictor=False, use_generator=False, neg_idxs=None, round=0):
        """
        Updates the labeled indices in the dataset.

        Args:
            pos_idxs (ndarray): The indices of positively labeled samples.
            neg_idxs (ndarray or None): The indices of negatively labeled samples. Defaults to None.
        """
        self.dataset.labeled_idxs[pos_idxs] = True
        states_path = "/root/Master_Thesis/scripts/notebooks/"
        if start_sam:
            self.sam_failed = []
            for idx in pos_idxs:
                model_predicted_masks = []
                sam_predicted_masks = []
                if not os.path.isfile(self.dataset.df["oracle"][idx]):                                               
          
                    handler = self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"])
                        
                    model_predicted_masks = self.getVotes(10, self.params, handler, round, pos_idxs)
                    current_mask = self.predict(self.dataset.handler([self.dataset.df["images"][idx]], [self.dataset.df["masks"][idx]], img_size=self.params["img_size"]))[0]
                    model_predicted_masks.append((current_mask.squeeze().cpu().sigmoid()> 0.5).float())
                    
                    for mask in model_predicted_masks:                            
                        boxes = self.sam.get_boxes(mask)
    
                        if len(boxes)>200:
                            boxes = boxes[:200]
        
                        sam_predicted_masks.append(self.sam.get_mask(img_path=self.dataset.df["images"][idx], boxes=boxes))
                    
                    for i in range(10):
                        sam_predicted_masks[i] = 0.06 * sam_predicted_masks[i]
                    
                    sam_predicted_masks[-1] = 0.4 * sam_predicted_masks[-1]
                    np_sam = np.array(sam_predicted_masks).sum(axis=0)
                    # print(np_sam.max(), np_sam.min())
                    majority = np.array((np_sam.squeeze() > 0.55), dtype=np.float32)
                    
                    # path = self.dataset.df["oracle"][idx].split("/")
                    # path[-2] = f'oracle_wmv_{self.params["img_size"][0]}_{round}'
                    # parent_dir = "/".join(path[:-1])
                    # path = "/".join(path)
                    # if not os.path.exists(parent_dir):
                    #     os.makedirs(parent_dir)
                    # np.save(path, majority.squeeze())
                    np.save(self.dataset.df["oracle"][idx], majority.squeeze())
    
        else: 
            self.human_envolved = len(pos_idxs)
        # if neg_idx return the path in 'oracle' to "empty"
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False
        
        return []
    
    def generatorSelection(self, sam_generated_masks):
        threshold = len(sam_generated_masks)/2
        sum_masks = np.array(sam_generated_masks).sum(axis=0)
        majority = np.array((sum_masks.squeeze() > threshold), dtype=np.float32)
        return majority

    
    
    def train(self):
        """
        Trains the network using the labeled data in the dataset.
        """
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        """
        Performs prediction using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The predicted labels.
            ndarray: The ground truth labels.
        """
        preds, masks_gt = self.net.predict(data)
        return preds, masks_gt

    def predict_prob(self, data):
        """
        Calculates the probability predictions using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        """
        Calculates the probability predictions using dropout in the network.

        Args:
            data: The input data.
            n_drop (int): The number of dropout iterations. Defaults to 10.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        """
        Calculates the probability predictions using dropout with split in the network.

        Args:
            data: The input data.
            n_drop (int): The number of dropout iterations. Defaults to 10.

        Returns:
            ndarray: The predicted probabilities.
        """
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        """
        Calculates the embeddings using the network.

        Args:
            data: The input data.

        Returns:
            ndarray: The calculated embeddings.
        """
        embeddings = self.net.get_embeddings(data)
        return embeddings
    

class RandomSampling(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle):
    # def __init__(self, dataset, net):
        """
        Initializes the RandomSampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(RandomSampling, self).__init__(dataset, net, sam)

    def query(self, n):
        """
        Selects a subset of unlabeled samples randomly for labeling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        return np.random.choice(np.where(self.dataset.labeled_idxs == 0)[0], n, replace=False)


class EntropySampling(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle):
    # def __init__(self, dataset, net):
        """
        Initializes the EntropySampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(EntropySampling, self).__init__(dataset, net, sam)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on entropy sampling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).view(len(unlabeled_idxs), -1).sum(1)
        top_n_idx = unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
        return top_n_idx
    
class MarginSampling(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle, params=None):
    # def __init__(self, dataset, net):
        """
        Initializes the MarginSampling strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(MarginSampling, self).__init__(dataset, net, sam, params)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on margin sampling.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs = probs.reshape(len(unlabeled_idxs), -1)
        max_probabilities, _ = probs.max(dim=1)
        min_probabilities, _ = probs.min(dim=1)
        uncertainties = max_probabilities - min_probabilities
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
    
class BALDDropout(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle, n_drop=10):
    # def __init__(self, dataset, net, n_drop=10):
        """
        Initializes the BALDDropout strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
            n_drop (int): The number of dropout iterations. Defaults to 10.
        """
        super(BALDDropout, self).__init__(dataset, net, sam)
        self.n_drop = n_drop

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on BALD dropout.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        pb = probs.mean(1)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
    
class AdversarialBIM(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle, eps=0.05):
    # def __init__(self, dataset, net, eps=0.05):
        """
        Initializes the AdversarialBIM strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
            eps (float): The epsilon value for the attack. Defaults to 0.05.
        """
        super(AdversarialBIM, self).__init__(dataset, net, sam)
        self.eps = eps

    def cal_dis(self, x):
        """
        Calculates the adversarial distance for a given sample.

        Args:
            x: The input sample.

        Returns:
            float: The adversarial distance.
        """
        nx = torch.unsqueeze(x, 0).cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).cuda()

        out = self.net.clf(nx + eta)
        mask_pred = torch.sigmoid(out)
        ny = mask_pred.round()

        while not torch.equal(mask_pred, ny):
            loss = F.binary_cross_entropy_with_logits(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out = self.net.clf(nx + eta)
            mask_pred = torch.sigmoid(out)
            ny = mask_pred.round()

        return (eta * eta).sum()

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on adversarial BIM.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm.tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.clf.train()

        return unlabeled_idxs[dis.argsort()[:n]]
    
class KCenterGreedy(Strategy):
    def __init__(self, dataset, net, sam:SAMOracle):
    # def __init__(self, dataset, net):
        """
        Initializes the KCenterGreedy strategy.

        Args:
            dataset: The dataset object.
            net: The network model.
        """
        super(KCenterGreedy, self).__init__(dataset, net, sam)

    def query(self, n):
        """
        Selects a subset of unlabeled samples based on K-Center Greedy algorithm.

        Args:
            n (int): The number of samples to query.

        Returns:
            ndarray: The indices of the selected samples.
        """
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm.tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

