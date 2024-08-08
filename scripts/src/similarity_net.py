import torch.nn as nn
import torch 
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class SimEmbeddings(nn.Module):
    def __init__(self, inp_dim):
        super(SimEmbeddings, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #After Conv
        # new_shape = ((old_shape−Kernal_Size+2Padding)/Stride)+1.
        inp_dim-=4
        #After Maxpooling
        #new_shape = old_shape/pooling_size
        inp_dim/=2
        #Flatten
        #64 out channels of the last conv layer
        inp_dim*=inp_dim*64
        self.fc1 = nn.Linear(int(inp_dim), 128)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        return x


class SimNet:
    def __init__(self, sim_model, params):
        self.sim_model = sim_model
        self.params = params
        self.optimizer = optim.SGD(list(self.sim_model.parameters()),lr = 1e-3, momentum=0.9)
        self.loss_fn = torch.nn.TripletMarginLoss(margin=3.0)

    def get_similarity(self, masks, gt_masks):
        iou = torch.zeros((masks.shape[0],1))
        for i in range(len(masks)):
            tp, fp, fn, tn = smp.metrics.get_stats(masks[i].long(), gt_masks[i].long(), mode="binary", threshold=0.5)
            iou[i] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou
        
        # iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        # return iou.sum(axis=1)/masks.shape[1]

    def train(self, data):
        losses = []
        epoch_loss=[]
        n_epoch = self.params['n_epoch']
        self.sim_model = self.sim_model.to("cuda")
        self.sim_model.train()
        
        loader = DataLoader(data, shuffle=True, batch_size=32)
        prog_bar = tqdm.tqdm(range(1, 150 + 1), ncols = 100, disable = False)
        for epoch in prog_bar:
            # for batch_idx, (x, y) in enumerate(loader):
            for batch_idx, (anchor, pos, neg) in enumerate(loader):
            
                # x, y = x.cuda(), y.cuda()
                anchor, pos, neg = anchor.cuda(), pos.cuda(), neg.cuda()
                self.optimizer.zero_grad()

                # predicted_similarity = self.sim_model(x).squeeze()
                anchor = self.sim_model(anchor)
                pos = self.sim_model(pos)
                neg = self.sim_model(neg)
                # gt_similarity = self.get_similarity(x,y).squeeze()
                # loss = self.loss_fn(predicted_similarity, gt_similarity)                
                loss = self.loss_fn(anchor, pos, neg) 
                loss.backward()
                epoch_loss.append(loss.item())
                self.optimizer.step()
                
                prog_bar.set_postfix(loss=loss.item())
            losses.append(sum(epoch_loss)/len(epoch_loss))
            epoch_loss=[]
        return losses

    def predict(self, mask_path):
        self.sim_model = self.sim_model.to("cuda")
        self.sim_model.eval()
        mask = torch.Tensor(np.load(mask_path, allow_pickle=True)).view(1,self.params["img_size"][0],self.params["img_size"][1]).cuda()
        return self.sim_model(mask)

    def get_embeddings(self, masks):
        self.sim_model = self.sim_model.to("cuda")
        self.sim_model.eval()
        return self.sim_model(masks)
        


class CustomImageDataset(Dataset):
    def __init__(self, comp_df, transform=None, target_transform=None):
        self.comp_df = comp_df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.comp_df)

    def __getitem__(self, index):
        # image = torch.Tensor(np.load(self.comp_df["mask"][index], allow_pickle=True))
        # label = torch.Tensor(np.load(self.comp_df["gt_mask"][index], allow_pickle=True))
        
        pos = torch.Tensor(np.load(self.comp_df["pos"][index], allow_pickle=True)).view(1,128,128)
        anchor = torch.Tensor(np.load(self.comp_df["anchor"][index], allow_pickle=True)).view(1,128,128)
        neg = torch.Tensor(np.load(self.comp_df["neg"][index], allow_pickle=True)).view(1,128,128)
        
        # return image, label
        return anchor, pos, neg