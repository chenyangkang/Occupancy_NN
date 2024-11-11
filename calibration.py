
import torch
import numpy as np 
from sklearn.linear_model import LogisticRegression
from torch import nn  
import torch.optim as optim
from loss import WeightedBCELoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  

class LogisticCallibrator(nn.Module):
    def __init__(self):
        super(LogisticCallibrator, self).__init__()
        self.slope = nn.Parameter(torch.tensor(1.0))  # Initialize slope parameter
        self.intercept = nn.Parameter(torch.tensor(0.0))  # Initialize intercept parameter

    def forward(self, x):
        # Logistic regression transformation with slope and intercept
        return torch.sigmoid(self.slope * x + self.intercept)

class NaivePredictor(nn.Module):
    def __init__(self):
        super(NaivePredictor, self).__init__()
        # Dummy parameters to allow requires_grad to be set
        self.dummy_weight = nn.Parameter(torch.tensor(1.0))
        self.dummy_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # Returns the input as-is, acting as an identity function
        return x

class ProbabilityCalibrator:
    """
	# logit (b,c,h,w): pre-softmax network output
	# beta (1,): user controlled hyperparameter 
	# s_prior (1,c): source (training) data prior
	# t_prior (1,c): target (test) data prior (most likely uniform)
    """
    def __init__(self, beta=1.0) -> None:
        self.beta=beta
        pass
    
    def prob_clip(self, prob):
        return torch.clip(prob, 1e-6, 1 - 1e-6) if isinstance(prob, torch.Tensor) else np.clip(prob, 1e-6, 1 - 1e-6)

    def fit(self, network, X_train, y_train, detect_vars, occupancy_vars, num_epochs=100000, lr=0.01, patience=20, min_delta=0.0, validation_split=0.2):
        self.s_prior = np.array([sum(y_train == 0) / len(y_train), sum(y_train == 1) / len(y_train)])
        self.t_prior = None

        # Split into training and validation sets
        if sum(y_train==1) > (1 / validation_split):
            # Enough data for splitting
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train)
        else:
            X_train, X_val, y_train, y_val = X_train, X_train, y_train, y_train
            
        X_det_train = X_train[detect_vars]
        X_occ_train = X_train[occupancy_vars]
        X_det_val = X_val[detect_vars]
        X_occ_val = X_val[occupancy_vars]

        with torch.no_grad():
            pred_det_train = network.predict_detection_probability(torch.from_numpy(np.array(X_det_train).astype('float32')))
            pred_det_train = self.prob_clip(pred_det_train).reshape(-1, 1)
            pred_det_train = torch.tensor(self.rebalancing(pred_det_train).reshape(-1, 1), dtype=torch.float32)
            pred_occ_train = network.predict_occupancy_probability(torch.from_numpy(np.array(X_occ_train).astype('float32')))
            pred_occ_train = self.prob_clip(pred_occ_train).reshape(-1, 1)
            pred_occ_train = torch.tensor(self.rebalancing(pred_occ_train).reshape(-1, 1), dtype=torch.float32)
            y_train_tensor = torch.from_numpy(y_train).float().reshape(-1, 1)
            y_train_tensor = self.prob_clip(y_train_tensor)

            pred_det_val = network.predict_detection_probability(torch.from_numpy(np.array(X_det_val).astype('float32')))
            pred_det_val = self.prob_clip(pred_det_val).reshape(-1, 1)
            pred_occ_val = network.predict_occupancy_probability(torch.from_numpy(np.array(X_occ_val).astype('float32')))
            pred_occ_val = self.prob_clip(pred_occ_val).reshape(-1, 1)
            y_val_tensor = torch.from_numpy(y_val).float().reshape(-1, 1)

        det_calibrator = LogisticCallibrator()
        occ_calibrator = LogisticCallibrator()

        optimizer = optim.Adam(list(det_calibrator.parameters()) + list(occ_calibrator.parameters()), lr=lr)
        criterion = WeightedBCELoss() #nn.BCELoss() #WeightedBCELoss()

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        improved = False

        for epoch in range(num_epochs):
            det_calibrator.train()
            occ_calibrator.train()
            optimizer.zero_grad()

            # Training predictions and loss
            calibrated_pred_det_train = det_calibrator(torch.log(pred_det_train / (1 - pred_det_train)))
            calibrated_pred_occ_train = occ_calibrator(torch.log(pred_occ_train / (1 - pred_occ_train)))
            combined_pred_train = calibrated_pred_det_train * calibrated_pred_occ_train
            combined_pred_train = self.prob_clip(combined_pred_train)
            train_loss = criterion(combined_pred_train[calibrated_pred_occ_train > 0.5], y_train_tensor[calibrated_pred_occ_train > 0.5])

            train_loss.backward()
            optimizer.step()

            # Validation predictions and loss
            det_calibrator.eval()
            occ_calibrator.eval()
            with torch.no_grad():
                calibrated_pred_det_val = det_calibrator(torch.log(pred_det_val / (1 - pred_det_val)))
                calibrated_pred_occ_val = occ_calibrator(torch.log(pred_occ_val / (1 - pred_occ_val)))
                combined_pred_val = calibrated_pred_det_val * calibrated_pred_occ_val
                combined_pred_val = self.prob_clip(combined_pred_val)
                val_loss = criterion(combined_pred_val, y_val_tensor)

            # Early stopping based on validation loss
            if val_loss.item() < best_val_loss - min_delta:
                best_val_loss = val_loss.item()
                best_det_calibrator = det_calibrator.state_dict()
                best_occ_calibrator = occ_calibrator.state_dict()
                epochs_without_improvement = 0
                improved = True
            else:
                epochs_without_improvement += 1

            # Check for early stopping
            if epochs_without_improvement >= patience:
                # print("Early stopping: no improvement after {} epochs.".format(patience))
                break

        if improved:
            # Load the best model parameters if there was an improvement
            self.det_calibrator = LogisticCallibrator()
            self.det_calibrator.load_state_dict(best_det_calibrator)
            self.occ_calibrator = LogisticCallibrator()
            self.occ_calibrator.load_state_dict(best_occ_calibrator)
        else:
            # No improvement, return original probabilities
            self.det_calibrator = NaivePredictor()
            self.occ_calibrator = NaivePredictor()

        return self

    # def predict(self, prob):
    #     prob = self.rebalancing(prob)
        
    def predict_det(self, prob):
        prob = self.rebalancing(prob)
        prob = self.prob_clip(prob)
        
        with torch.no_grad():
            prob = torch.from_numpy(prob).float().reshape(-1,1)
            logit = torch.log(prob/(1-prob))
            prob = self.det_calibrator(logit).cpu().numpy()
        return prob
    
    def predict_occ(self, prob):
        prob = self.rebalancing(prob)
        prob = self.prob_clip(prob)
        
        with torch.no_grad():
            prob = torch.from_numpy(prob).float().reshape(-1,1)
            logit = torch.log(prob/(1-prob))
            prob = self.occ_calibrator(logit).cpu().numpy()
        return prob
    
    def rebalancing(self, prob):
        prob = prob.reshape(-1,1)
        prob = self.prob_clip(prob)
        
        prob = np.concatenate([(1-prob).reshape(-1,1), prob.reshape(-1,1)], axis=1)
        inv_prior = self.s_prior #1/self.s_prior
        # inv_prior[inv_prior == float("inf")] = 0
        inv_prior = self.prob_clip(inv_prior).reshape(1,-1)

        if self.t_prior is None:
            prob_r = prob*inv_prior
        else:
            prob_r = prob*inv_prior*self.t_prior

        # Normalize each row by its sum
        prob_r = prob_r / (np.sum(prob_r, axis=1))[:,np.newaxis]
        
        outputs = prob**(1-self.beta) * prob_r**self.beta
        outputs = outputs / (np.sum(outputs, axis=1))[:,np.newaxis]
        return outputs[:,1]
