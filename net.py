import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from skorch.callbacks import EpochScoring
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import OneCycleLR
from skorch.callbacks import EarlyStopping
from sklearn.calibration import CalibratedClassifierCV
from skorch.dataset import ValidSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F
import pickle 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, make_scorer
from functools import partial
from sklearn.isotonic import IsotonicRegression


# Combined Model for Detection and Occupancy
class CombinedModel(nn.Module):
    def __init__(self, input_dim_det, input_dim_occ, latent_size_det=8, latent_size_occ=64, latent_layer_det=2, latent_layer_occ=2):
        super(CombinedModel, self).__init__()
        
        # Detection Sub-network
        self.det_layers = nn.ModuleList()
        self.det_layers.append(nn.Linear(input_dim_det, latent_size_det))
        self.det_layers.append(nn.BatchNorm1d(latent_size_det))
        self.det_layers.append(nn.ReLU())
        self.det_layers.append(nn.Dropout(0.3))
        for _ in range(latent_layer_det - 1):
            self.det_layers.append(nn.Linear(latent_size_det, latent_size_det))
            self.det_layers.append(nn.BatchNorm1d(latent_size_det))
            self.det_layers.append(nn.ReLU())
            self.det_layers.append(nn.Dropout(0.3))
        self.fc_det_out = nn.Linear(latent_size_det, 1)
        
        # Occupancy Sub-network
        self.occ_layers = nn.ModuleList()
        self.occ_layers.append(nn.Linear(input_dim_occ, latent_size_occ))
        self.occ_layers.append(nn.BatchNorm1d(latent_size_occ))
        self.occ_layers.append(nn.ReLU())
        self.occ_layers.append(nn.Dropout(0.3))
        for _ in range(latent_layer_occ - 1):
            self.occ_layers.append(nn.Linear(latent_size_occ, latent_size_occ))
            self.occ_layers.append(nn.BatchNorm1d(latent_size_occ))
            self.occ_layers.append(nn.ReLU())
            self.occ_layers.append(nn.Dropout(0.3))
        self.fc_occ_out = nn.Linear(latent_size_occ, 1)
        
        # Dropout layer applied after the last hidden layer
        self.dropout_det = nn.Dropout(0.3)
        self.dropout_occ = nn.Dropout(0.3)

        self.step = 0
        
        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_det, X_occ):
        self.step += 1
        det_prob = self.predict_detection_probability(X_det)
        occ_prob = self.predict_occupancy_probability(X_occ)
        observation_outcome = occ_prob * det_prob
        return observation_outcome
    
    def predict_detection_probability(self, X_det):
        # Detection Pathway with dynamic layers
        x_det = X_det
        for layer in self.det_layers:
            x_det = layer(x_det)
        x_det = self.dropout_det(x_det) 
        x_det = self.fc_det_out(x_det)
        det_prob = self.sigmoid(x_det)  # Detection probability
        det_prob = torch.clip(det_prob, 1e-6, 1 - 1e-6)
        return det_prob
    
    def predict_occupancy_probability(self, X_occ):
        # Occupancy Pathway with dynamic layers
        x_occ = X_occ
        for layer in self.occ_layers:
            x_occ = layer(x_occ)
        x_occ = self.dropout_occ(x_occ)
        x_occ = self.fc_occ_out(x_occ)
        occ_prob = self.sigmoid(x_occ)  # Occupancy probability
        occ_prob = torch.clip(occ_prob, 1e-6, 1 - 1e-6)
        return occ_prob