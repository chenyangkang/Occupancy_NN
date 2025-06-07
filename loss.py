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

class occ_loss(nn.Module):
    """ N-mixture loss.
    
    Args:
      y_obs (tensor): nsite by nrep count observation matrix
      psi_hat (tensor): occupancy expected value
      p_hat (tensor): individual detection probability
    
    Returns:
      negative log-likelihood (tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, y_obs):
        psi_hat, p_hat = output
        batch_size, n_rep = y_obs.shape
        z_vals = torch.tensor([0.0, 1.0]).view(1, 2) # possible_z_vals
        z_logprob = torch.distributions.Bernoulli(probs=psi_hat).log_prob(z_vals) # (batch_size, 2)
        
        all_zero = (y_obs.sum(dim=1) == 0)       # shape = (batch_size,), bool
        inf = float('inf')
        loglik_y_given_z0 = torch.where(all_zero, 
                                        torch.zeros(batch_size),  # log(1)=0
                                        -inf * torch.ones(batch_size))

        log_p    = torch.log(p_hat.clamp(min=1e-8))       # → (batch_size,)
        log_1mp  = torch.log((1-p_hat).clamp(min=1e-8))   # → (batch_size,)
        log_p_expand   = log_p.view(batch_size, 1)       # → (batch_size, 1)
        log_1mp_expand = log_1mp.view(batch_size, 1)     # → (batch_size, 1)
        term_rep = y_obs * log_p_expand + (1 - y_obs) * log_1mp_expand
        loglik_y_given_z1 = term_rep.sum(dim=1)  
        
        y_logprob = torch.stack(
            [loglik_y_given_z0,    # (batch_size,)
            loglik_y_given_z1],   # (batch_size,)
            dim=1                   # → (batch_size, 2)
        )

        joint_logprob = z_logprob + y_logprob   # shape = (batch_size, 2)
        log_lik_per_site = torch.logsumexp(joint_logprob, dim=1)  # → (batch_size,)

        return -log_lik_per_site.sum()



class FocalLoss(nn.Module):

    def __init__(self, gamma = 1.0):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype = torch.float32)
        self.eps = 1e-6

    def forward(self, input, target):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid
        target = target.float()
        weight = self.calculate_sample_weights(target)
        self.alpha=weight
        
        BCE_loss = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
            
    def calculate_sample_weights(self, target_tensor):
        
        # Calculate class frequencies within the batch
        positive_count = target_tensor.sum()
        negative_count = target_tensor.size(0) - positive_count

        # Avoid division by zero
        weight_pos = (1 / positive_count).item() if positive_count > 0 else 0
        weight_neg = (1 / negative_count).item() if negative_count > 0 else 0

        # Apply weights to the loss based on class
        # weight = torch.tensor(weight_pos/weight_neg)
        weight = torch.where(target_tensor == 1, weight_pos, weight_neg).view(-1,1)
        return weight
        
        

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        # self.bce_loss = nn.BCELoss(reduction='mean')  # Don't reduce yet to apply weights

    def forward(self, outputs, targets):
        # Ensure targets are float for BCE Loss
        targets = targets.float()

        # Calculate class frequencies within the batch
        positive_count = targets.sum()
        negative_count = targets.size(0) - positive_count

        # Avoid division by zero
        weight_pos = (1 / positive_count).item() if positive_count > 0 else 0
        weight_neg = (1 / negative_count).item() if negative_count > 0 else 0

        # Apply weights to the loss based on class
        weights = torch.where(targets == 1, weight_pos, weight_neg).view(-1,1)
        BCE_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        weighted_loss = (weights * BCE_loss).mean()

        return weighted_loss# weighted_loss


class UnweightedBCELoss(nn.Module):
    def __init__(self):
        super(UnweightedBCELoss, self).__init__()
        # self.bce_loss = nn.BCELoss(reduction='mean')  # Don't reduce yet to apply weights

    def forward(self, outputs, targets):
        # Ensure targets are float for BCE Loss
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        unweighted_loss = BCE_loss.mean()

        return unweighted_loss

class DeferredWeightedBCELoss(nn.Module):
    def __init__(self):
        super(DeferredWeightedBCELoss, self).__init__()
        # self.bce_loss = nn.BCELoss(reduction='mean')  # Don't reduce yet to apply weights
        self.step = 0
        self.shifted = False
        
    def forward(self, outputs, targets):
        self.step += 1
        

        # Ensure targets are float for BCE Loss
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        
        if self.step <= 10000:
            weighted_loss = BCE_loss.mean()
        else:
            if not self.shifted:
                print('Shifting to weighted BCELoss!')
                self.shifted = True
            else:
                pass
            # Calculate class frequencies within the batch
            positive_count = targets.sum()
            negative_count = targets.size(0) - positive_count

            # Avoid division by zero
            weight_pos = (1 / positive_count).item() if positive_count > 0 else 0
            weight_neg = (1 / negative_count).item() if negative_count > 0 else 0

            # Apply weights to the loss based on class
            weights = torch.where(targets == 1, weight_pos, weight_neg).view(-1,1)
            weighted_loss = (weights * BCE_loss).mean()

        return weighted_loss# weighted_loss

