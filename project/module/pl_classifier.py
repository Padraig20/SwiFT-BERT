import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics import  PearsonCorrCoef # Accuracy,
from torchmetrics.regression import R2Score
from sklearn.metrics import balanced_accuracy_score, roc_curve
import monai.transforms as monai_t
from .utils.lr_scheduler import CosineAnnealingWarmUpRestarts

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb


from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool

from einops import rearrange

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LitClassifier(pl.LightningModule):
    def __init__(self,data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
        
        if self.hparams.label_scaling_method == 'standardization':
            target_values = data_module.train_dataset.target_values
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(normalized_target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax': 
            target_values = data_module.train_dataset.target_values
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler

        self.model = load_model("swin4d_ver7", self.hparams)
        
        # HEAD
        self.output_head = load_model(self.hparams.head, self.hparams)

        self.metric = Metrics()

        if self.hparams.adjust_thresh:
            self.threshold = 0

    def forward(self, x):
        return self.output_head(self.model(x)) # (b, t, e) = [16, 20, 7]
    
    def augment(self, img):
        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            rotate_range=(0.175, 0.175, 0.175), # 0.175 rad = 10 degrees
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) # 

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            for t in range(T): # set augmentation seed to be the same for all time steps
                comp.set_random_state(seed=aug_seed)
                img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])
        img = rearrange(img, 'b t c h w d -> b c h w d t')    
        return img
    
    def save_encoder(self, path):
        torch.save(self.model.swinViT.state_dict(), f"{path}/SwiFT-encoder.pth")
    
    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training:
            fmri = self.augment(fmri)

        feature = self.model(fmri)

        if self.hparams.downstream_task == 'emotion' or self.hparams.downstream_task == 'emotionDM':
            feature = self.model(fmri)
            logits = self.output_head(feature) 
            target = target_value.float() # (b,t,e)
        # Classification task
        elif self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            logits = self.output_head(feature).squeeze() #self.clf(feature).squeeze()
            target = target_value.float().squeeze()
        # Regression task
        elif self.hparams.downstream_task_type == 'regression':
            feature = self.model(fmri)
            # target_mean, target_std = self.determine_target_mean_std()
            logits = self.output_head(feature) # (batch,1) or # tuple((batch,1), (batch,1))
            unnormalized_target = target_value.float() # (batch,1)
            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        
        return subj, logits, target
    
    def _calculate_loss(self, batch, batch_idx, mode):
        subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training) # (batch, T, E)

        if 'emotion' in self.hparams.downstream_task and self.hparams.downstream_task_type == 'regression':

            if not self.hparams.evaluate_separately:
                logits = logits.view(logits.size(0), -1)  # (batch_size, T * E)
                target = target.view(target.size(0), -1)  # (batch_size, T * E)

                if self.hparams.loss_type == 'mse':
                    loss = F.mse_loss(logits, target)
                elif self.hparams.loss_type == 'mae':
                    loss = F.l1_loss(logits, target)

                within_subj_loss = loss

                mse = F.mse_loss(logits, target)
                l1 = F.l1_loss(logits, target)

                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_mse": mse, 
                    f"{mode}_l1_loss": l1,
                    f"{mode}_within_subj_loss": within_subj_loss.item(),
                }
            else:
                result_dict = {}
                for i in range(self.hparams.target_dim):
                    #logits_group = logits.view(logits.size(0), -1)[i::self.hparams.target_dim]  # (batch_size, T * E)
                    #target_group = target.view(target.size(0), -1)[i::self.hparams.target_dim]  # (batch_size, T * E)

                    logits_group = logits[..., i]
                    target_group = target[..., i]
                    
                    if self.hparams.loss_type == 'mse':
                        loss = F.mse_loss(logits_group, target_group)
                    elif self.hparams.loss_type == 'mae':
                        loss = F.l1_loss(logits_group, target_group)

                    within_subj_loss = loss

                    mse = F.mse_loss(logits_group, target_group)
                    l1 = F.l1_loss(logits_group, target_group)
                    
                    result_dict.update({
                    f"{mode}_loss_{i}": loss.item(),
                    f"{mode}_mse_{i}": mse.item(),
                    f"{mode}_l1_loss_{i}": l1.item(),
                    f"{mode}_within_subj_loss_{i}": within_subj_loss.item()
                    })
                logits = logits.view(logits.size(0), -1)  # (batch_size, T * E)
                target = target.view(target.size(0), -1)  # (batch_size, T * E)

                if self.hparams.loss_type == 'mse':
                    loss = F.mse_loss(logits, target)
                elif self.hparams.loss_type == 'mae':
                    loss = F.l1_loss(logits, target)
                    
                result_dict.update({
                    f"{mode}_loss": loss.item(),
                })
        
        elif 'emotion' in self.hparams.downstream_task and self.hparams.downstream_task_type == 'classification':
            
            if not self.hparams.evaluate_separately:
            
                logits = logits.view(logits.size(0), -1)  # (batch_size, T * E)
                target = target.view(target.size(0), -1)  # (batch_size, T * E)
            
                loss = F.binary_cross_entropy_with_logits(logits, target) # target is float
                acc = self.metric.get_accuracy_binary(logits, target)
            
                result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_acc": acc,
                }
            else:
                result_dict = {}
                #print(logits.shape, target.shape)
                #print(logits.view(logits.size(0), -1).shape, target.view(target.size(0), -1).shape)
                for i in range(self.hparams.target_dim):
                    #logits_group = logits.view(logits.size(0), -1)[i::self.hparams.target_dim]  # (batch_size, T * E)
                    #target_group = target.view(target.size(0), -1)[i::self.hparams.target_dim]  # (batch_size, T * E)
                    
                    logits_group = logits[..., i]  # Shape: [batch_size, temporal_size]
                    target_group = target[..., i]
                    
                    #print(i, logits_group.shape, target_group.shape)

                    loss = F.binary_cross_entropy_with_logits(logits_group, target_group)  # target is float

                    acc = self.metric.get_accuracy_binary(logits_group, target_group)

                    result_dict.update({
                    f"{mode}_loss_{i}": loss.item(),
                    f"{mode}_acc_{i}": acc.item()
                    })
                    
                logits = logits.view(logits.size(0), -1)  # (batch_size, T * E)
                target = target.view(target.size(0), -1)  # (batch_size, T * E)
                loss = F.binary_cross_entropy_with_logits(logits, target) # target is float
                result_dict.update({
                    f"{mode}_loss": loss.item(),
                })
            
        elif self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            loss = F.binary_cross_entropy_with_logits(logits, target) # target is float
            acc = self.metric.get_accuracy_binary(logits, target.float().squeeze())
            result_dict = {
            f"{mode}_loss": loss,
            f"{mode}_acc": acc,
            }

        elif self.hparams.downstream_task_type == 'regression':
            loss = F.mse_loss(logits.squeeze(), target.squeeze())
            l1 = F.l1_loss(logits.squeeze(), target.squeeze())
            result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_mse": loss,
                f"{mode}_l1_loss": l1
            }
            
        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size) # batch_size = batch_size
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode):
        
        """
        Evaluates the model's performance on an emotion task, where targets are 2D vectors (T x E).
    
        Parameters:
        - total_out: The output tensor of the model containing predictions and targets.
        - subj_array: Array indicating the subject corresponding to each sample.
        - subjects: List of unique subjects.
        - scaler: Scaler object used for label scaling (e.g., standardization or minmax).
        - mode: The mode of the evaluation (e.g., 'train', 'val', 'test').
        """
        
        subjects = np.unique(subj_array)
        
        subj_avg_logits = []
        subj_targets = []
        
        # print(total_out.shape) # [4, 2, 25, 7] -> s, b, t, e
    
        # do not calculate the average logits, but flatten logits and targets and calculate metrics
        for subj in subjects:
            subj_logits = total_out[subj_array == subj, 0]  # shape: [T, E]
            subj_target = total_out[subj_array == subj, 1]  # shape: [T, E]
                        
            #print(subj_logits.shape, subj_target.shape)
        
            #avg_logits = torch.mean(subj_logits, dim=0)  # average over emotions E: shape: [T, E] -> [E]
            #avg_target = torch.mean(subj_target, dim=0)
            
            avg_logits = torch.flatten(subj_logits) # shape: [T, E] -> [T*E]
            avg_target = torch.flatten(subj_target)
            
            #print(avg_logits.shape, avg_target.shape)
        
            subj_avg_logits.append(avg_logits)
            subj_targets.append(avg_target)
    
        # lists -> tensors
        
        #print("HALLO")
        #print(subj_avg_logits)
        #print(subj_targets)
        #print("HALLO")
        
        subj_avg_logits = torch.stack(subj_avg_logits).to(total_out.device).squeeze()  # Shape: [num_subjects, E]
        subj_targets = torch.stack(subj_targets).to(total_out.device).squeeze()  # Shape: [num_subjects, E]
        
        #print("HALLO")
        #print(subj_avg_logits.shape, subj_targets.shape)
        #print("HALLO")
        
        subj_avg_logits = subj_avg_logits.flatten().squeeze()
        subj_targets = subj_targets.flatten().squeeze()
    
        if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            if self.hparams.adjust_thresh:
                # move threshold to maximize balanced accuracy
                best_bal_acc = 0
                best_thresh = 0
                for thresh in np.arange(-5, 5, 0.01):
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=thresh).int().cpu())
                    if bal_acc > best_bal_acc:
                        best_bal_acc = bal_acc
                        best_thresh = thresh
                self.log(f"{mode}_best_thresh", best_thresh, sync_dist=True)
                self.log(f"{mode}_best_balacc", best_bal_acc, sync_dist=True)
                fpr, tpr, thresholds = roc_curve(subj_targets.cpu(), subj_avg_logits.cpu())
                idx = np.argmax(tpr - fpr)
                youden_thresh = thresholds[idx]
                acc_func = BinaryAccuracy().to(total_out.device)
                self.log(f"{mode}_youden_thresh", youden_thresh, sync_dist=True)
                self.log(f"{mode}_youden_balacc", balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=youden_thresh).int().cpu()), sync_dist=True)

                if mode == 'valid':
                    self.threshold = youden_thresh
                elif mode == 'test':
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=self.threshold).int().cpu())
                    self.log(f"{mode}_balacc_from_valid_thresh", bal_acc, sync_dist=True)
            else:
                acc_func = BinaryAccuracy().to(total_out.device)
            
            auroc_func = BinaryAUROC().to(total_out.device)
                
            #auroc_func = BinaryAUROC().to(total_out.device)
            #print(subj_avg_logits)
            #print(subj_targets)
            acc = acc_func((subj_avg_logits >= 0).int(), subj_targets)
            #print((subj_avg_logits>=0).int().cpu())
            #print(subj_targets.cpu())
            bal_acc_sk = balanced_accuracy_score((subj_avg_logits>=0).int().cpu(), subj_targets.cpu())
            auroc = auroc_func((subj_avg_logits >= 0).int().cpu(), subj_targets.cpu())

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc_sk, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

            if self.hparams.evaluate_separately:
                for i in range(self.hparams.target_dim):
                    preds_group = subj_avg_logits[i::self.hparams.target_dim]
                    targets_group = subj_targets[i::self.hparams.target_dim]
                    
                    #preds_group = subj_avg_logits[..., i]
                    #targets_group = subj_targets[..., i]
                    
                    #print(subj_avg_logits.shape, subj_targets.shape)
                    #print(
                    #    f"preds_group: {preds_group.shape}, targets_group: {targets_group.shape}"
                    #)
    
                    preds_group_binary = (preds_group >= 0).int()
                    #targets_group_binary = (targets_group >= 0).int()
                    
                    #print(preds_group_binary)
                    #print(targets_group_binary)
    
                    acc = acc_func(preds_group_binary, targets_group)
    
                    bal_acc_sk = balanced_accuracy_score(preds_group_binary.cpu(), targets_group.cpu())
    
                    auroc = auroc_func((preds_group >= 0).int().cpu(), targets_group.cpu())
                        
                    self.log(f"{mode}_acc_{i}", acc, sync_dist=True)
                    self.log(f"{mode}_balacc_{i}", bal_acc_sk, sync_dist=True)
                    self.log(f"{mode}_AUROC_{i}", auroc, sync_dist=True)
            
        else: # regression
            
            pearson = PearsonCorrCoef().to(total_out.device)    
            r2score = R2Score().to(total_out.device)
            
            # losses
            mse = F.mse_loss(subj_avg_logits, subj_targets)
            mae = F.l1_loss(subj_avg_logits, subj_targets)
    
            # reconstruct to original scale if necessary
            if self.hparams.label_scaling_method == 'standardization': # default
                scale = self.scaler.scale_
                mean = self.scaler.mean_
                adjusted_mse = F.mse_loss(subj_avg_logits * scale + mean, subj_targets * scale + mean)
                adjusted_mae = F.l1_loss(subj_avg_logits * scale + mean, subj_targets * scale + mean)
            elif self.hparams.label_scaling_method == 'minmax':
                data_max = self.scaler.data_max_
                data_min = self.scaler.data_min_
                adjusted_mse = F.mse_loss(subj_avg_logits * (data_max - data_min) + data_min, subj_targets * (data_max - data_min) + data_min)
                adjusted_mae = F.l1_loss(subj_avg_logits * (data_max - data_min) + data_min, subj_targets * (data_max - data_min) + data_min)
            else:
                adjusted_mse = mse
                adjusted_mae = mae

            pearson_coef = pearson(subj_avg_logits.flatten(), subj_targets.flatten())
            r2_output = r2score(subj_avg_logits.flatten(), subj_targets.flatten())

            self.log(f"{mode}_corrcoef", pearson_coef, sync_dist=True)
            self.log(f"{mode}_r2", r2_output, sync_dist=True)
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)
            self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True)
            self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True)
                
            if self.hparams.evaluate_separately:
                for i in range(self.hparams.target_dim):
                    preds_group = subj_avg_logits[i::self.hparams.target_dim]
                    targets_group = subj_targets[i::self.hparams.target_dim]

                    mse = F.mse_loss(preds_group, targets_group)
                    mae = F.l1_loss(preds_group, targets_group)

                    # reconstruct to original scale if necessary
                    if self.hparams.label_scaling_method == 'standardization':  # default
                        scale = self.scaler.scale_
                        mean = self.scaler.mean_
                        adjusted_preds = preds_group * scale + mean
                        adjusted_targets = targets_group * scale + mean
                        adjusted_mse = F.mse_loss(adjusted_preds, adjusted_targets)
                        adjusted_mae = F.l1_loss(adjusted_preds, adjusted_targets)
                    elif self.hparams.label_scaling_method == 'minmax':
                        data_max = self.scaler.data_max_
                        data_min = self.scaler.data_min_
                        adjusted_preds = preds_group * (data_max - data_min) + data_min
                        adjusted_targets = targets_group * (data_max - data_min) + data_min
                        adjusted_mse = F.mse_loss(adjusted_preds, adjusted_targets)
                        adjusted_mae = F.l1_loss(adjusted_preds, adjusted_targets)
                    else:
                        adjusted_mse = mse
                        adjusted_mae = mae

                    pearson_coef = pearson(preds_group.flatten(), targets_group.flatten())
                    r2_output = r2score(preds_group.flatten(), targets_group.flatten())

                    self.log(f"{mode}_corrcoef_{i}", pearson_coef, sync_dist=True)
                    self.log(f"{mode}_r2_{i}", r2_output, sync_dist=True)
                    self.log(f"{mode}_mse_{i}", mse, sync_dist=True)
                    self.log(f"{mode}_mae_{i}", mae, sync_dist=True)
                    self.log(f"{mode}_adjusted_mse_{i}", adjusted_mse, sync_dist=True)
                    self.log(f"{mode}_adjusted_mae_{i}", adjusted_mae, sync_dist=True)


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, batch_idx, mode="train") 
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.hparams.downstream_task == 'emotion':
            subj, logits, target = self._compute_logits(batch)
            output = torch.stack([logits, target], dim=-1)
            return (subj, output.detach().cpu())
        else:
            subj, logits, target = self._compute_logits(batch)
            output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)
            return (subj, output.detach().cpu())

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        outputs_valid = outputs[0]
        outputs_test = outputs[1]
        subj_valid = []
        subj_test = []
        out_valid_list = []
        out_test_list = []
        for subj, out in outputs_valid:
            subj_valid += subj
            out_valid_list.append(out)
        for subj, out in outputs_test:
            subj_test += subj
            out_test_list.append(out)
        subj_valid = np.array(subj_valid)
        subj_test = np.array(subj_test)
        total_out_valid = torch.cat(out_valid_list, dim=0)
        total_out_test = torch.cat(out_test_list, dim=0)
        
        # save only if necessary for further analysis
        #self._save_predictions(subj_valid,total_out_valid,mode="valid")
        #self._save_predictions(subj_test,total_out_test, mode="test") 

        # evaluate 
        self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
        self._evaluate_metrics(subj_test, total_out_test, mode="test")
            
    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1
        
        if self.hparams.strategy == None : 
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            if (world_size > 1) and (len(self.subject_accuracy) > 0):
                total_subj_accuracy = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks     
                accuracy_dict = {}
                for dct in total_subj_accuracy:
                    for subj, metric_dict in dct.items():
                        if subj not in accuracy_dict:
                            accuracy_dict[subj] = metric_dict
                        else:
                            accuracy_dict[subj]['score']+=metric_dict['score']
                            accuracy_dict[subj]['count']+=metric_dict['count']
                self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved) 
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)


    def _save_predicted_map_and_target(self, subj_array, total_out, mode):
        # print('total_out.device',total_out.device)
        # (total iteration/world_size) numbers of samples are passed into _evaluate_metrics.
        subjects = np.unique(subj_array)

        # subj_sex = []
        subj_avg_logits = np.empty((len(subjects), total_out.shape[1])) 
        subj_targets = np.empty((len(subjects), total_out.shape[1])) 
        for idx, subj in enumerate(subjects):
            #print('total_out.shape:',total_out.shape) # torch.Size([32, 132032, 2])
            subj_logits = total_out[subj_array == subj,:,0]
            subj_avg_logits[idx,:] = torch.mean(subj_logits, axis=0).detach().cpu().numpy() # average predicted task maps of the specific subject
            subj_targets[idx,:] = total_out[subj_array == subj,:,1][0,:].detach().cpu().numpy()
        
        #current_rank = torch.distributed.get_rank()
        if self.trainer.is_global_zero:
            os.makedirs(os.path.join('predictions',self.hparams.id),exist_ok=True)
            with open(os.path.join('predictions',self.hparams.id,f'test_subj_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
                pickle.dump(subjects, pickle_out)
                
            with open(os.path.join('predictions',self.hparams.id,f'predicted_map_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
                pickle.dump(subj_avg_logits, pickle_out)

            with open(os.path.join('predictions',self.hparams.id,f'target_map_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
                pickle.dump(subj_targets, pickle_out)
    
    def test_step(self, batch, batch_idx):
        # do nothing for test step since val step also performs test step
        if self.hparams.downstream_task == 'emotion':
            subj, logits, target = self._compute_logits(batch)
            output = torch.stack([logits, target], dim=-1)
            return (subj, output.detach().cpu())
        else:
            subj, logits, target = self._compute_logits(batch)
            output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)
            return (subj, output)

    def test_epoch_end(self, outputs):
        subj_test = [] 
        out_test_list = []
        for subj, out in outputs:
            subj_test += subj
            out_test_list.append(out.detach())
        subj_test = np.array(subj_test)
        total_out_test = torch.cat(out_test_list, dim=0)
        
        #self._save_predictions(subj_test, total_out_test, mode="test") 
        self._evaluate_metrics(subj_test, total_out_test, mode="test")

    def on_test_epoch_start(self) -> None:
        return super().on_test_epoch_start()
    
    def on_train_epoch_start(self) -> None:
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings=np.zeros((self.repetitions,1))
        return super().on_train_epoch_start()
    
    def on_train_batch_start(self, batch, batch_idx):
        torch.cuda.nvtx.range_push("train") 
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx-self.gpu_warmup] = curr_time
            elif (batch_idx-self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)
                
                Throughput = (self.repetitions*self.hparams.batch_size*int(self.hparams.num_nodes) * int(self.hparams.devices))/self.total_time
                
                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:',mean_syn)
                print('std_syn:',std_syn)
                
        return super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        torch.cuda.nvtx.range_pop() # train
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        torch.cuda.nvtx.range_push("valid")
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        torch.cuda.nvtx.range_pop()
        return super().on_validation_epoch_end()

    def on_before_backward(self, loss: torch.Tensor) -> None:
        torch.cuda.nvtx.range_push("backward")
        return super().on_before_backward(loss)

    def on_after_backward(self) -> None:
        torch.cuda.nvtx.range_pop()
        return super().on_after_backward()

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")
        
        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            warmup = int(total_iterations * self.hparams.warmup) #?
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * self.hparams.warmup) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 2 #? 1 in SwiFUN
            
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        ## training related
        # group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        # group.set_defaults(grad_clip=False)

        group.add_argument("--loss_type", type=str, default="mse", help="which loss to use. You can use reconstructive-contrastive loss with 'rc'")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.set_defaults(use_scheduler=False)
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--warmup", type=float, default=0.01, help="warmup in CosineAnnealingWarmUpRestarts (recommend 0.01~0.1 values)")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=0.5, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")
        
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.set_defaults(augment_during_training=False)
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        
        ## model related
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--out_chans", type=int, default=1, help="Channel size of target output")
        group.add_argument("--mlp_dim", type=int, default=512, help="hidden dimension of MLP head")
        group.add_argument("--target_dim", type=int, default=5, help="dimension of target output, e.g. number of emotions per timeframe")
        group.add_argument("--embed_dim", type=int, default=36, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", type=int, default=[4, 4, 4, 6], help="window size from the second layers")
        group.add_argument("--patch_size", nargs="+", type=int, default=[6, 6, 6, 1], help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage of encoder")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")
        group.add_argument("--first_window_size", nargs="+", type=int, default=[4, 4, 4, 6], help="window size in the first layer")
        
        group.add_argument("--head", type=str, default="linear", help="architecture for decoder head, choose from: linear, mlp, bert, lstm")
        group.add_argument("--lstm_dim", type=int, default=256, help="hidden dimension of LSTM head")
        group.add_argument("--lstm_layers", type=int, default=2, help="number of layers for LSTM head")
        
        group.add_argument("--evaluate_separately", action='store_true', help="whether to evaluate each emotion separately")
        
        ## BERT
        group.add_argument("--bert_num_layers", type=int, default=6, help="number of layers in BERT")
        group.add_argument("--bert_num_heads", type=int, default=8, help="number of heads in BERT")
        group.add_argument("--bert_intermediate_size", type=int, default=2048, help="intermediate size in BERT")
        group.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1, help="hidden dropout probability in BERT")
        group.add_argument("--bert_attention_probs_dropout_prob", type=float, default=0.1, help="attention dropout probability in BERT")
        group.add_argument("--bert_vocab_size", type=int, default=30522, help="vocab size in BERT")
        group.add_argument("--bert_pretrained_model_name", type=str, default=None, help="pretrained model name in BERT, if wanted")

        ## others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")        
        return parser
