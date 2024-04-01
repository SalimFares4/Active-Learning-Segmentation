#!/usr/bin/env python

"""models.py contains Pytorch models"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


import torch.nn as nn
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from transformers.modeling_outputs import SemanticSegmenterOutput





class Net:
    def __init__(self, net, params, device):
        """
        Initializes the Net class.

        Args:
            net: The neural network model.
            params: Dictionary of parameters.
            device: Device to run the model on.
        """
        self.net = net
        self.params = params
        self.device = device
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.losses=[]
    def train(self, data):
        """
        Trains the neural network model.

        Args:
            data: The training dataset.
        """
        n_epoch = self.params['n_epoch']
        self.clf = self.net.to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        prog_bar = tqdm.tqdm(range(1, n_epoch + 1), ncols=100)
        for epoch in prog_bar:
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                    
                if type(out) is SemanticSegmenterOutput:
                    out = out.logits
                    # print("out_shape: ", out.shape)
                    # print("y_shape: ", y.shape)
                    out = nn.functional.interpolate(out,
                                                size=y.shape[-2:],
                                                mode="bilinear",
                                                align_corners=False,
                                            ).argmax(dim=1)
                    out = out.type(torch.FloatTensor).requires_grad_()
                    out = out.to(self.device)
                # print("out_shape: ", out.shape)
                # print("y_shape: ", y.shape)
                
                loss = self.loss_fn(out, y)
                self.losses.append(loss)
                loss.backward()
                optimizer.step()
                # prog_bar.set_postfix({"Dice_loss": loss.item()})
                prog_bar.set_postfix(loss =  loss.item())
    
    
    def predict(self, data):
        """
        Generates predictions using the trained model.

        Args:
            data: The input dataset for prediction.

        Returns:
            preds: Predicted masks.
            masks: Ground truth masks.
        """
        self.clf.eval()
        preds = torch.zeros(len(data), 256, 256)
        masks = torch.zeros(len(data), 256, 256)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                    
                if type(out) is SemanticSegmenterOutput:
                    out = out.logits
                    out = nn.functional.interpolate(out,
                                                size=y.shape[-2:],
                                                mode="bilinear",
                                                align_corners=False,
                                            ).argmax(dim=1)
                    out = out.type(torch.FloatTensor)#.requires_grad_()
                # print(out.shape)
                # print(preds.shape)
                preds[idxs] = out.squeeze().cpu()
                masks[idxs] = y.squeeze().cpu()
        # if masks.shape == (1024, 1, 1024):
        #     print(104)
        return preds, masks

    def predict_prob(self, data):
        """
        Generates probability predictions using the trained model.

        Args:
            data: The input dataset for prediction.

        Returns:
            probs: Predicted probabilities.
        """
        self.clf.eval()
        probs = torch.zeros(len(data), 256, 256)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                if type(out) is SemanticSegmenterOutput:
                    out = out.logits
                    out = nn.functional.interpolate(out,
                                                size=y.shape[-2:],
                                                mode="bilinear",
                                                align_corners=False,
                                            ).argmax(dim=1)
                    out = out.type(torch.FloatTensor)#.requires_grad_()
                prob = F.sigmoid(out)
                probs[idxs] = prob.squeeze().cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        """
        Generates probability predictions using dropout with the trained model.

        Args:
            data: The input dataset for prediction.
            n_drop: Number of dropout iterations.

        Returns:
            probs: Predicted probabilities.
        """
        self.clf.train()
        probs = torch.zeros(len(data), 256, 256)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    if type(out) is SemanticSegmenterOutput:
                        out = out.logits
                        out = nn.functional.interpolate(out,
                                                    size=y.shape[-2:],
                                                    mode="bilinear",
                                                    align_corners=False,
                                                ).argmax(dim=1)
                        out = out.type(torch.FloatTensor)#.requires_grad_()
                    prob = F.sigmoid(out)#.view(1, -1)
                    print(out.shape)
                    probs[idxs] += prob.cpu()
                    
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        """
        Generates probability predictions using dropout with the trained model.

        Args:
            data: The input dataset for prediction.
            n_drop: Number of dropout iterations.

        Returns:
            probs: Predicted probabilities.
        """
        self.clf.train()
        probs = torch.zeros([n_drop, len(data)])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    if type(out) is SemanticSegmenterOutput:
                        out = out.logits
                        out = nn.functional.interpolate(out,
                                                    size=y.shape[-2:],
                                                    mode="bilinear",
                                                    align_corners=False,
                                                ).argmax(dim=1)
                        out = out.type(torch.FloatTensor)#.requires_grad_()
                    prob = F.sigmoid(out).view(-1)
                    probs[i][idxs] += prob.cpu()
        return probs

    def get_embeddings(self, data):
        """
        Retrieves embeddings from the trained model.

        Args:
            data: The input dataset.

        Returns:
            embeddings: Extracted embeddings.
        """
        self.clf.eval()
        embeddings = torch.zeros([len(data), 512 * 20 * 60])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf.encoder(x)[5]
                embeddings[idxs] = out.view(-1).cpu()
        return embeddings


class Segmentor(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": dataset_f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)






class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsample layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Output layers
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.output(x)
        # Upsample output to (N, 3, 1024, 1024)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
