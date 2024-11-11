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
from sklearn.base import BaseEstimator

from loss import FocalLoss, WeightedBCELoss, UnweightedBCELoss, DeferredWeightedBCELoss
from calibration import ProbabilityCalibrator
from net import CombinedModel

import copy

config = {
    'loss': WeightedBCELoss,#DeferredWeightedBCELoss, #WeightedBCELoss,#nn.BCELoss,#FocalLoss,
}


def my_scoring(model_, X, y, metric='roc_auc'):
    
    pred_proba = model_.predict_proba(X)
    if len(pred_proba.shape)==2:
        if pred_proba.shape[1]==2:
            pred_proba = pred_proba[:,1].flatten()
    else:
        pred_proba = pred_proba.flatten()
    
    if metric=='roc_auc':
        return roc_auc_score(y, pred_proba)
    elif metric=='f1':
        return f1_score(y, np.where(pred_proba>0.5, 1, 0))
    elif metric=='recall':
        return recall_score(y, np.where(pred_proba>0.5, 1, 0))
    elif metric=='precision':
        return precision_score(y, np.where(pred_proba>0.5, 1, 0))
    else:
        raise
    
        
class occupancy_ml_trainer(BaseEstimator):
    def __init__(self, batch_size=128, max_epochs=1000, 
                 latent_size_det=8, latent_layer_det=2,
                 latent_size_occ=64, latent_layer_occ=2,
                 verbose=1, 
                 no_mini_batch=False, validation=False, 
                 tolerance_epoch=5, tolerance_threashold=0, 
                 scoring='roc_auc',
                 val_split = 0.1,
                 balance_sampling=False,
                 probability_calibration=True,
                 do_early_stopping=True,
                 partial_fitting=True) -> None:
        
        self.batch_size = batch_size
        self.no_mini_batch = no_mini_batch
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.validation = validation
        self.tolerance_epoch = tolerance_epoch
        self.tolerance_threashold = tolerance_threashold
        self.scoring = scoring
        self.latent_size_det=latent_size_det
        self.latent_size_occ=latent_size_occ
        self.latent_layer_det = latent_layer_det
        self.latent_layer_occ = latent_layer_occ
        self.val_split = val_split
        self.balance_sampling = balance_sampling
        self.probability_calibration = probability_calibration
        self.do_early_stopping = do_early_stopping
        self.X_detection_var_normalizer = MinMaxScaler()
        self.X_occupancy_var_normalizer = MinMaxScaler()
        self.model = None
        self.calibrator = None
        self.partial_fitting = partial_fitting

    def upsampler(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, np.where(y>0, 1, 0))
        return X, y


    def fit(self, X_train, y_train):
        
        y_train = np.where(y_train>0, 1, 0)
        
        if self.probability_calibration:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_split, stratify=y_train, random_state=42
            )
        
        if self.balance_sampling:
            X_train, y_train = self.upsampler(X_train, y_train)
        
        self.detect_vars = [i for i in X_train.columns if i.startswith('detect_')]
        self.occupancy_vars = [i for i in X_train.columns if not i in self.detect_vars]
        self.detect_var_size = len(self.detect_vars)
        self.occupancy_var_size = len(self.occupancy_vars)
        
        X_train_detection_var_df = X_train[self.detect_vars]
        X_train_detection_var_df = self.X_detection_var_normalizer.fit_transform(X_train_detection_var_df)
        X_train_occupancy_var_df = X_train[self.occupancy_vars]
        X_train_occupancy_var_df = self.X_occupancy_var_normalizer.fit_transform(X_train_occupancy_var_df)
        
        if self.no_mini_batch:
            self.batch_size = X_train.shape[0]
            
        self.lr_scheduler = LRScheduler(
            policy=OneCycleLR,
            max_lr=0.01,  # max learning rate for the cycle
            # steps_per_epoch=len(X_train_detection_var_df) // self.batch_size,  # batch size is 64
            # epochs=self.max_epochs
        )
            
        callbacks = [EpochScoring(scoring=partial(my_scoring, metric='f1'),lower_is_better=False,name=f'train_f1', on_train=True),
                       EpochScoring(scoring=partial(my_scoring, metric='roc_auc'),lower_is_better=False,name=f'train_roc_auc', on_train=True),
                       EpochScoring(scoring=partial(my_scoring, metric='f1'),lower_is_better=False,name=f'valid_f1', on_train=False),
                       EpochScoring(scoring=partial(my_scoring, metric='roc_auc'),lower_is_better=False,name=f'valid_roc_auc', on_train=False),
                       EpochScoring(scoring=partial(my_scoring, metric='recall'),lower_is_better=False,name=f'valid_recall', on_train=False),
                       EpochScoring(scoring=partial(my_scoring, metric='precision'),lower_is_better=False,name=f'valid_precision', on_train=False),
                       ]
        if self.do_early_stopping:
            self.early_stopping = EarlyStopping(
                monitor=f'valid_{self.scoring}' if self.validation else f'train_{self.scoring}',  # Monitor the validation AUC score
                patience=self.tolerance_epoch,          # Number of epochs with no improvement to wait before stopping
                threshold=self.tolerance_threashold,      # Minimum change to consider an improvement
                threshold_mode='rel', # Use a relative change (0.1% improvement) as the threshold
                lower_is_better=False # Higher AUC is better
            )
            callbacks.append(self.early_stopping)

        # Wrap the model in skorch's NeuralNetClassifier
        if self.model is None:
            self.model = NeuralNetClassifier(
                CombinedModel,
                module__input_dim_det=X_train_detection_var_df.shape[1],
                module__input_dim_occ=X_train_occupancy_var_df.shape[1],
                module__latent_size_det=self.latent_size_det,
                module__latent_size_occ=self.latent_size_occ,
                module__latent_layer_det=self.latent_layer_det,
                module__latent_layer_occ=self.latent_layer_occ,
                criterion= config['loss'](),#FocalLoss, #WeightedBCELoss, #nn.BCELoss(), #WeightedBCELoss,#nn.BCELoss,#WeightedBCELoss,#nn.BCELoss ,#WeightedBCELoss,
                optimizer=optim.Adam,
                max_epochs=self.max_epochs,
                lr=0.01,
                batch_size=self.batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(cv=5, stratified=True) if self.validation else None,
                callbacks=callbacks,
                verbose=self.verbose,
                # warmstart=True
            )

        # Train the model
        # model.fit(X_train_detection_var_df.astype('float32'), X_train_occupancy_var_df.astype('float32'), y_train.astype('float32'))
        if self.partial_fitting:
            self.model.batch_size = self.batch_size
            self.model.partial_fit(
                {"X_det": torch.tensor(np.array(X_train_detection_var_df), dtype=torch.float32), 
                "X_occ": torch.tensor(np.array(X_train_occupancy_var_df), dtype=torch.float32)},
                torch.tensor(np.array(np.where(y_train>0, 1, 0)).reshape(-1,1), dtype=torch.float32)
            )
        else:
            self.model = copy.deepcopy(self.model)
            self.model.fit(
                {"X_det": torch.tensor(np.array(X_train_detection_var_df), dtype=torch.float32), 
                "X_occ": torch.tensor(np.array(X_train_occupancy_var_df), dtype=torch.float32)},
                torch.tensor(np.array(np.where(y_train>0, 1, 0)).reshape(-1,1), dtype=torch.float32)
            )
        
        # Fine-tune temperatures using the split-off validation set
        if self.probability_calibration:
            # if self.calibrator is None:
            self.calibrator = ProbabilityCalibrator().fit(self.model.module_, X_val, y_val, self.detect_vars, self.occupancy_vars)
            # else:
            #     self.calibrator.fit(self.model.module_, X_val, y_val, self.detect_vars, self.occupancy_vars)
                
        return self
    
    def predict_detection_probability(self, X_det):
        
        if not X_det.shape[1] == self.detect_var_size:
            raise ValueError(f'Input predictor shape is different from training data!')
        
        for col in X_det.columns:
            if not col in self.detect_vars:
                raise ValueError(f'{col} not in self.detect_vars!')
            
        X_det = self.X_detection_var_normalizer.transform(X_det)
            
        with torch.no_grad():
            pred = self.model.module_.predict_detection_probability(torch.from_numpy(np.array(X_det).astype('float32')))
            
        pred = pred.detach().cpu().numpy().reshape(-1,1)
                
        if self.probability_calibration:
            pred = self.calibrator.predict_det(pred)
                        
        pred = np.concatenate([(1-pred).reshape(-1,1), pred.reshape(-1,1)], axis=1)
        return pred
    
    def predict_occupancy_probability(self, X_occ):
        
        if not X_occ.shape[1] == self.occupancy_var_size:
            raise ValueError(f'Input predictor shape is different from training data!')
        
        X_occ = self.X_occupancy_var_normalizer.transform(X_occ)
        
        with torch.no_grad():
            pred = self.model.module_.predict_occupancy_probability(torch.from_numpy(np.array(X_occ).astype('float32')))
            
        pred = pred.detach().cpu().numpy().reshape(-1,1)
                
        if self.probability_calibration:
            pred = self.calibrator.predict_occ(pred)
                    
        pred = np.concatenate([(1-pred).reshape(-1,1), pred.reshape(-1,1)], axis=1)
        return pred
    
    def predict_proba(self, X, stage='combined'):
        """Predicting probability
        
        Args:
            X:
                Prediciton set
            stage:
                One of 'combined', 'detection', or 'occupancy'.
        """
        
        X_detection_var_df = X[self.detect_vars]
        X_occupancy_var_df = X[self.occupancy_vars]
        
        if stage == 'combined':
            proba = self.predict_detection_probability(X_detection_var_df) * self.predict_occupancy_probability(X_occupancy_var_df)
        elif stage == 'detection':
            proba = self.predict_detection_probability(X_detection_var_df)
        elif stage == 'occupancy':
            proba = self.predict_occupancy_probability(X_occupancy_var_df)
        else:
            raise ValueError(f"stage must be one of 'combined', 'detection', or 'occupancy'.")
        
        # proba[:,0] = 1-proba[:,1].flatten()
        proba = np.concatenate([(1-proba[:,1]).reshape(-1,1), (proba[:,1]).reshape(-1,1)], axis=1)
        return proba

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        return np.where(pred_proba[:,1]>0.5, 1, 0)
    
    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path, fine_tuning=False):
        with open(path,'rb') as f:
            self = pickle.load(f)
        
        if fine_tuning:

            for param in self.model.module_.parameters():
                param.requires_grad = False

            # Unfreeze only the last layers for fine-tuning
            for param in self.model.module_.fc_det_out.parameters():
                param.requires_grad = True
            for param in self.model.module_.fc_occ_out.parameters():
                param.requires_grad = True
                
            for param in self.calibrator.det_calibrator.parameters():
                param.requires_grad = True
                
            for param in self.calibrator.occ_calibrator.parameters():
                param.requires_grad = True

        
        return self