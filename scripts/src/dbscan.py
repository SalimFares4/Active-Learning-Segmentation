import numpy as np
import torch
import segmentation_models_pytorch as smp


class DBScan():
    def __init__(self, similarity, eps = 0.3, min_samples=4):
        self.similarity = similarity
        self.eps=eps
        self.min_samples = min_samples


    
    def distance(self, x1, x2):
        return 1 - self.similarity(x1, x2)
        
    def fit(self, masks):
       
        n_samples = masks.shape[0]
        labels = torch.zeros(n_samples, dtype=torch.int)
     
        # Initialize cluster label and visited flags
        cluster_label = 0
        visited = torch.zeros(n_samples, dtype=torch.bool)
     
        # Iterate over each point
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
     
            # Find neighbors
            neighbors = torch.nonzero(torch.tensor(np.array([self.distance(masks[i], masks[j]) < self.eps for j in range(len(masks))])))
            if neighbors.shape[0] < self.min_samples:
                # Label as noise
                labels[i] = 0
            else:
                # Expand cluster
                cluster_label += 1
                labels[i] = cluster_label
                labels = self.expand_cluster(masks, labels, visited, neighbors, cluster_label)
     
        return torch.nonzero(labels).flatten().tolist()

    def expand_cluster(self, masks, labels, visited, neighbors, cluster_label):
        i = 0
        while i < neighbors.shape[0]:
            neighbor_index = neighbors[i].item()
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = torch.nonzero(torch.tensor(np.array([self.distance(masks[neighbor_index], masks[j]) < self.eps for j in range(len(masks))])))
                if neighbor_neighbors.shape[0] >= self.min_samples:
                    neighbors = torch.cat((neighbors, neighbor_neighbors))
            if labels[neighbor_index] == 0:
                labels[neighbor_index] = cluster_label
            i += 1
        return labels

class Similarities():

    def __init__(self):
        pass
    
    def iou_score(self, x, y):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        prob_mask = x.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Compute true positive, false positive, false negative, and true negative 'pixels' for each class
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), y.long(), mode="binary")
        # Calculate IoU
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou

    def cosine_similarity(self, x, y):
        return np.dot(x.flatten(), y.flatten()) / (np.sqrt(np.dot(x.flatten(), x.flatten())) * np.sqrt(np.dot(y.flatten(), y.flatten())))

    def eculidian_distance(self, x, y):
        return 1 - (self.l2_normalize(x) - self.l2_normalize(y)).pow(2).sum().sqrt()
        
    def l2_normalize(self, v):
        norm = np.sqrt(np.square(v).sum())
        return v / norm

        