import numpy as np
import torch
import segmentation_models_pytorch as smp
import cv2
from scipy.spatial.distance import cdist

class DBScan():
    def __init__(self, similarities, eps = 0.3, min_samples=4):
        self.similarities = similarities
        self.eps=eps
        self.min_samples = min_samples


    
    def distance(self, x1, x2):
        distance = 0
        for similarity in self.similarities:
            distance+= 1 - similarity(x1, x2)
        return distance/float(len(self.similarities))
        
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
            neighbors = torch.nonzero(torch.tensor(np.array([1.0 if i==j else self.distance(masks[i], masks[j]) < self.eps for j in range(len(masks))])))
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
        x = x.float().flatten()
        y = y.float().flatten()
        return 0.5 * ((np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))) + 1)

    def eculidian_distance(self, x, y):
        return 1 - (self.l2_normalize(x) - self.l2_normalize(y)).pow(2).sum().sqrt()
        
    def l2_normalize(self, v):
        norm = np.sqrt(np.square(v).sum())
        return v / norm


class Gestalt():
    
    def __init__(self):
        pass

    def find_contours(self, mask):
        if torch.is_tensor(mask):
            mask = mask.numpy()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    
    def find_centroids(self, mask):
        contours, _ = self.find_contours(mask)
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centroids.append((cX, cY))
        return np.array(centroids)

    def proximity(self, mask1, mask2):
        centroids1 = self.find_centroids(mask1)
        centroids2 = self.find_centroids(mask2)
        
        if len(centroids1) == 0 or len(centroids2) == 0:
            return float('inf')  # No centroids found in one of the masks
        
        distances = cdist(centroids1, centroids2, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        average_distance = np.mean(min_distances)
    
        max_distance = np.sqrt(mask1.shape[0]**2 + mask1.shape[1]**2)
        
        similarity = 1 - average_distance / max_distance
        
        return similarity

    def count_closures(self, contours, hierarchy):
        closed_contours = 0
        for i in range(len(contours)):
            opened = hierarchy[0][i][2]<0 and hierarchy[0][i][3]<0            
            if not opened:
                closed_contours += 1
        return closed_contours

    def closure(self, mask1, mask2):
        contours1, hierarchy1  = self.find_contours(mask1)
        contours2, hierarchy2  = self.find_contours(mask2)
        
        closed_contours1 = self.count_closures(contours1, hierarchy1)
        closed_contours2 = self.count_closures(contours2, hierarchy2)
        
        # Calculate the closure similarity
        total_contours = max(len(contours1), len(contours2))
        if total_contours == 0:
            return 1.0  # No contours in either mask, consider it perfect closure similarity
        
        closure_distance = abs(closed_contours1 - closed_contours2) / total_contours
        closure_similarity = 1 - closure_distance  # Higher value means more similar
        
        return closure_similarity