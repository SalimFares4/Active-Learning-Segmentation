import torch
import cv2
import torch.optim as optim
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import tqdm
from PIL import Image
import numpy as np

# class PromptEmbeddings(torch.nn.Module):
#     def __init__(self, sparse_in_channels, sparse_out_channels, dense_in_channels, dense_out_channels):
#         super(PromptEmbeddings, self).__init__()
#         self.sparse_layer = torch.nn.Linear(sparse_in_channels, sparse_out_channels)
#         self.dense_layer = torch.nn.Linear(dense_in_channels, dense_out_channels)
        
#     def forward(self, x):
#         coords = x.reshape(-1, 2, 2).clone()
#         coords.requires_grad_(True)
        
#         learnable_sparse_embeddings = self.sparse_layer(coords)
        
#         learnable_dense_embeddings = self.dense_layer.weight.reshape(1, -1, 1, 1).expand(x.shape[0], -1, 64, 64)
#         learnable_dense_embeddings.requires_grad_(True)

#         return learnable_sparse_embeddings, learnable_dense_embeddings

class PromptEmbeddings(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PromptEmbeddings, self).__init__()
        self.learnable_weights = torch.nn.Linear(64, 64, bias=False)
        # self.learnable_embeddings = torch.nn.Linear(in_channels, out_channels)
        self.learnable_embeddings = torch.nn.Linear(4096, 256)
        # self.learnable_embeddings = torch.rand(256, 64, 64,requires_grad = True).cuda()
        
    def forward(self, x):
        # C = x.shape[1]
        # Z_hat_T = torch.rand(C, 64, 64,requires_grad = True).cuda()
        # # Z_T_default = torch.rand(256, 64, requires_grad = True)
        # W_hat_T = torch.rand(C, requires_grad = True).cuda()
        # W_hat_T = W_hat_T.view(C, 1, 1)

        # Z_T_SemPrompt = W_hat_T * Z_hat_T + (1 - W_hat_T) * x
        # #Z_T_SemPrompt = Z_hat_T * W_hat_T.view(C, 1) +  x * (1 - W_hat_T.view(C, 1))
        # return Z_T_SemPrompt
    
         # embeddings = self.learnable_weights(self.learnable_embeddings.weight.reshape(1, -1, 1, 1).expand(x.shape[0], -1, 64, 64)) + x - self.learnable_weights(x)
        # embeddings = self.learnable_weights(self.learnable_embeddings) + x - self.learnable_weights(x)
        embeddings = self.learnable_weights(self.learnable_embeddings.weight.view(256, 64, 64)) + x - self.learnable_weights(x)
        # embeddings = self.learnable_weights(x)
        # embeddings = x * self.learnable_weights
        
        # embeddings.requires_grad_(True)

        return embeddings


class PromptNet:
    def __init__(self, model, sam, params, df):
        self.clf = model
        self.sam = sam
        self.params = params
        self.df = df
        for param in self.sam.mask_predictor.model.mask_decoder.parameters():
            param.requires_grad = False
            
    def train(self):
        self.clf = self.clf.to("cuda")
        self.clf.train()
        # optimizer = optim.SGD(list(self.clf.parameters()) + list(self.sam.mask_predictor.model.mask_decoder.parameters()),lr = 1e-5, momentum=0.9)
        # optimizer = optim.SGD(list(self.sam.mask_predictor.model.mask_decoder.parameters()),lr = 1e-5, momentum=0.9)
        
        # optimizer = optim.SGD([{'params': self.clf.parameters(), 'lr': 5e-3},
                                # {'params':self.sam.mask_predictor.model.mask_decoder.parameters(), 'lr': 1e-5}], momentum=0.9)
        # optimizer = optim.SGD(list(self.sam.mask_predictor.model.mask_decoder.parameters()),lr = 1e-5, momentum=0.9)
        
        optimizer = optim.SGD(list(self.clf.parameters()),lr = 1e-5, momentum=0.9)
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits = True)
        prog_bar = tqdm.tqdm(range(1, self.params["n_epoch"] + 1), ncols = 100, disable = False)
    
        for epoch in prog_bar:
            for idx in range(len(self.df)):
                mask = cv2.resize(np.load(self.df["masks"][idx], allow_pickle=True), (256, 256), interpolation=cv2.INTER_CUBIC)
                boxes = self.sam.get_boxes(mask)
                if len(boxes) < 1:
                    continue
                boxes = torch.Tensor(boxes).to("cuda")
                boxes = self.sam.mask_predictor.transform.apply_boxes_torch(boxes, self.params["img_size"])
                sparse_embeddings, dense_embeddings = self.sam.mask_predictor.model.prompt_encoder(points=None, boxes=boxes, masks=None)
                optimizer.zero_grad()
                
                dense_embeddings = dense_embeddings.detach().clone()
                # learnable_sparse_embeddings, learnable_dense_embeddings = self.clf(boxes)
                
                # _, learnable_dense_embeddings = self.clf(boxes)
                
                # # sparse_embeddings = learnable_sparse_embeddings * (sparse_embeddings * (1 - self.clf.sparse_layer.weight.T))
                
                # dense_embeddings = learnable_dense_embeddings * (dense_embeddings * (1 - self.clf.dense_layer.weight.reshape(1, -1, 1, 1).expand(boxes.shape[0], -1, 64, 64)))

                dense_embeddings = self.clf(dense_embeddings)
                
                img_rgb = cv2.resize(np.load(self.df["images"][idx], allow_pickle=True), (256, 256), interpolation=cv2.INTER_CUBIC)
                
                self.sam.mask_predictor.set_image(img_rgb)
                
                logits, iou_predictions = self.sam.mask_predictor.model.mask_decoder(
                    image_embeddings=self.sam.mask_predictor.features,
                    image_pe=self.sam.mask_predictor.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                logits = logits.sum(axis=0).squeeze()
                y = torch.Tensor(mask).float().to("cuda")
                loss = loss_fn(logits, y)                
                # Print gradients before and after backward pass
                # for name, param in self.clf.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print(f"Grad before backward for {name}: {param.grad}")

                loss.backward(retain_graph=True)

                # torch.nn.utils.clip_grad_norm_(self.clf.parameters(), max_norm=1.0)
                
                # for name, param in self.clf.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print(f"Grad after backward for {name}: {param.grad}")

                optimizer.step()
                
                prog_bar.set_postfix(loss=loss.item())
        # return learnable_sparse_embeddings, learnable_dense_embeddings

    def predict(self, img_path, mask_path):
        self.clf = self.clf.to("cuda")
        self.clf.eval()
        mask = cv2.resize(np.load(mask_path, allow_pickle=True), (256, 256), interpolation=cv2.INTER_CUBIC)
        boxes = self.sam.get_boxes(mask)
        if len(boxes) < 1:
            print("No boxes found")
            return 0
        boxes = torch.Tensor(boxes).to("cuda")
        boxes = self.sam.mask_predictor.transform.apply_boxes_torch(boxes, self.params["img_size"])
        sparse_embeddings, dense_embeddings = self.sam.mask_predictor.model.prompt_encoder(points=None, boxes=boxes, masks=None)
        # learnable_sparse_embeddings, learnable_dense_embeddings = self.clf(boxes)
        # _, learnable_dense_embeddings = self.clf(boxes)
        
        # sparse_embeddings = learnable_sparse_embeddings * (sparse_embeddings * (1 - self.clf.sparse_layer.weight.T))
        
        # dense_embeddings = learnable_dense_embeddings * (dense_embeddings * (1 - self.clf.dense_layer.weight.reshape(1, -1, 1, 1).expand(boxes.shape[0], -1, 64, 64)))

        dense_embeddings = self.clf(dense_embeddings)
        
        img_rgb = cv2.resize(np.load(img_path, allow_pickle=True), (256, 256), interpolation=cv2.INTER_CUBIC)
        
        self.sam.mask_predictor.set_image(img_rgb)
        
        logits, iou_predictions = self.sam.mask_predictor.model.mask_decoder(
            image_embeddings=self.sam.mask_predictor.features,
            image_pe=self.sam.mask_predictor.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return (logits.sum(axis=0).squeeze().cpu().sigmoid()>0.5).float()
