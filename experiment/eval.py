import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from scipy.stats import spearmanr
from diffunc.utils import extract_substring
from diffunc.all import get_image_paths
from object_metric import default_instancer, segment_metrics, aggregate, relabel_connected_components, calculate_weighted #object wise metric

class PReval():
    def __init__(self, image_paths, save_path, mask_parent_path, result_save_path, cond_type, upsample_model, unc_type, 
                    do_filter = False, threshold = 300, weighted_auprc = False, auprc = True, obj = True, weighted_obj = False, obj_thresh_p = [0.3], obj_thresh_segsize = 500, 
                    iou_thresholds = np.linspace(0.25, 0.75, 11, endpoint=True), thresh_sIoU = np.linspace(0.25, 0.75, 11, endpoint=True),
                    obj_verbose  = True, overwrite = False, debug = False, baseline = False, sqr = False, psnr  =  True, image_size = (550, 688),
                    label_save_dir = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/labels/ood', freq_types = ['pixel', 'image'], rellis = False, portion  = 1):
        self.rellis = rellis
        self.image_paths = image_paths
        self.save_path = save_path
        self.mask_parent_path = mask_parent_path
        self.cond_type = cond_type
        self.upsample_model = upsample_model
        self.unc_type = unc_type
        self.do_filter = do_filter
        self.threshold = threshold
        self.auprc = auprc
        self.weighted_auprc = weighted_auprc
        # self.pixel = pixel # use pixel wise freq for weighted auprc
        self.psnr = psnr # psnr metric
        self.obj = obj
        self.weighted_obj = weighted_obj
        self.obj_thresh_p = obj_thresh_p
        self.obj_thresh_segsize = obj_thresh_segsize
        self.iou_thresholds = iou_thresholds
        self.thresh_sIoU = thresh_sIoU
        self.obj_verbose = obj_verbose
        self.overwrite = overwrite
        self.debug = debug
        self.freq_types = freq_types
        self.result_save_path = result_save_path
        self.pr_save_path = os.path.join(result_save_path, 'pr_results', f'{cond_type}_{upsample_model}_{unc_type}.pt')
        # self.weighted_pr_save_path = os.path.join(result_save_path, 'weighted_pr_results', f'{cond_type}_{upsample_model}_{unc_type}_{metric_type}.pt')
        self.metric_save_path = os.path.join(result_save_path, 'metrics', f'{cond_type}_{upsample_model}_{unc_type}.pt')
        # self.weighted_metric_save_path = os.path.join(result_save_path, 'weighted_metrics', f'{cond_type}_{upsample_model}_{unc_type}_{metric_type}.pt')
        self.obj_metric_save_path = os.path.join(result_save_path, f'obj_metrics_{int(obj_thresh_segsize)}', f'{cond_type}_{upsample_model}_{unc_type}.pt')
        self.weighted_pixel_obj_metric_save_path = os.path.join(result_save_path, f'weighted_pixel_obj_metrics_{int(obj_thresh_segsize)}', f'{cond_type}_{upsample_model}_{unc_type}.pt')
        self.weighted_img_obj_metric_save_path = os.path.join(result_save_path, f'weighted_img_obj_metrics_{int(obj_thresh_segsize)}', f'{cond_type}_{upsample_model}_{unc_type}.pt')
        self.baseline = baseline # eval for baseline
        self.sqr = sqr # sqr the unc
        self.label_save_dir = label_save_dir # dir of freq tensors for each image
        self.image_size = image_size
        self.portion = portion #how much of the image to take e.g. 8 -> take 1/64 of the image

    def calculate_weighted_aucpr(self, labels, uncertainties, weighted_pr_save_path, weighted_metric_save_path):
        '''
        labels: OOD-weights, [0, 1], larger, more uncertain
        '''
        # if already exists, return
        if not self.overwrite and glob.glob(weighted_metric_save_path) and glob.glob(weighted_pr_save_path):
            print('result already generated at', weighted_metric_save_path)
            metrics = torch.load(weighted_metric_save_path)
            return metrics

        # concatenate lists for labels and uncertainties together
        print('label shape', labels[0].shape)
        if (labels[0].shape[-1] > 1 and np.ndim(labels[0]) > 2) or \
                (labels[0].shape[-1] == 1 and np.ndim(labels[0]) > 3):
            # data is already in batches
            labels = np.concatenate(labels)
            uncertainties = np.concatenate(uncertainties)
        else:
            labels = np.stack(labels)
            uncertainties = np.stack(uncertainties)
        labels = labels.squeeze()
        uncertainties = uncertainties.squeeze()

        # NOW CALCULATE METRICS
        # pos = labels == 1 # ood, positive
        ''''changed '''
        # print('wegiths', labels)
        valid = labels != -1  # filter out void, which has weights hard coded to be -1 
        gt = labels[valid] 
        np.save('gt.npy', gt)
        print('total len', len(gt))
        gt_size = gt.size
        del gt, labels
       
        ''''end '''
        # del pos
        uncertainty = uncertainties[valid].reshape(-1).astype(np.float32, copy=False)
        np.save('uncertainty.npy', uncertainty)
        del valid, uncertainties

        print('start sorting')

        # Sort the classifier scores (uncertainties)
        sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
        uncertainty = uncertainty[sorted_indices]
        np.save('si.npy', sorted_indices)
        del sorted_indices
        print('uncertainty sorting finished')

        # Remove duplicates along the curve: remove the ones with the same unc
        distinct_value_indices = np.where(np.diff(uncertainty))[0] # get indices with that unc
    
        del uncertainty
        threshold_idxs = np.r_[distinct_value_indices, gt_size - 1] # add gt.size - 1 as the last element of distinct_value_indices
        del distinct_value_indices
        

        print('start gt')
        gt = np.load('gt.npy')
        sorted_indices =  np.load('si.npy')
        gt = gt[sorted_indices]
        print('sorted')
        # print('uncertainty', uncertainty)
        # print('gt', sum(gt[:3957]))
        del sorted_indices

        
        # Accumulate TPs and FPs
        tps = np.cumsum(gt)[threshold_idxs] # cumsum of ood weights at the diff unc levels
        # print('weights', gt[threshold_idxs])
        # print('tps', tps) #[::-1][:3])
        del gt
        ''''changed '''
        # fps = 1 + threshold_idxs - tps
        # print('fps', fps[::-1][:3])     
        # del threshold_idxs

        # Compute Precision and Recall
        precision = tps / (threshold_idxs + 1) # (tps + fps)
        # print('threshold_idxs', threshold_idxs)
        
        ''''end '''
        # precision must be decreasing: flip, take cum max, flip
        precision = np.flip(np.maximum.accumulate(np.flip(precision, axis = 0)), axis = 0) #precision.flip(0).cummax(0)[0].flip(0)
        precision[np.isnan(precision)] = 0
        
        recall = tps / tps[-1]
        print('total', tps[-1])
        # print('precision', precision) #[::-1][:5])
        # print('recall', recall) #[::-1][:5])
        # stop when full recall attained and reverse the outputs so recall is decreasing
        sl = slice(tps.searchsorted(tps[-1]), None, -1)
        precision = np.r_[precision[sl], 1]
        recall = np.r_[recall[sl], 0]
        average_precision = -np.sum(np.diff(recall) * precision[:-1])
        aucpr = -round(np.trapz(precision, recall),5)

        curve_prec, curve_recall = get_curve_pr(torch.from_numpy(precision), torch.from_numpy(recall))
        del precision, recall
        pr_ts = {
            'curve_precision': torch.from_numpy(np.array(curve_prec)),
            'curve_recall': torch.from_numpy(np.array(curve_recall)),
            'tps': tps,
            'threshold_idxs': threshold_idxs,
            'aucpr': aucpr
            }
        if not self.debug: torch.save(pr_ts, weighted_pr_save_path)
        del pr_ts
        # pr_ts = {
        #     'precision': torch.from_numpy(precision),
        #     'recall': torch.from_numpy(recall),
        #     'tps': tps,
        #     'threshold_idxs': threshold_idxs,
        #     'aucpr': aucpr
        #     }
        
        tpr = tps / tps[-1]
        fps = threshold_idxs+1 - tps
        del tps
        fpr = fps / fps[-1]
        del fps, threshold_idxs
        fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
        fpr_tpr90 = fpr[np.searchsorted(tpr, 0.90)]
        fpr_tpr85 = fpr[np.searchsorted(tpr, 0.85)]
        fpr_tpr80 = fpr[np.searchsorted(tpr, 0.80)]
        fpr_tpr75 = fpr[np.searchsorted(tpr, 0.75)]
        del tpr, fpr
        # pr_ts = pr_ts | {'fpr_tpr95': round(fpr_tpr95, 3),
        #     'fpr_tpr90': round(fpr_tpr90, 3),
        #     'fpr_tpr85': round(fpr_tpr85, 3),
        #     'fpr_tpr80': round(fpr_tpr80, 3),
        #     'fpr_tpr75': round(fpr_tpr75, 3)}

        


        # if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        #     # Add an extra threshold position if necessary
        #     # to make sure that the curve starts at (0, 0)
        #     tps = np.r_[0., tps]
        #     fps = np.r_[0., fps]

        # # Compute TPR and FPR
        # tpr = tps / tps[-1]
        # del tps
        # fpr = fps / fps[-1]
        # del fps

        print('AP:', average_precision)
        # del curve_precision, curve_recall
        # uncertainty = np.load('uncertainty.npy')
        # gt = np.load('gt.npy')
        # correlation, _ = spearmanr(uncertainty, gt)  
        # del gt, uncertainty
        # print("Spearman's rank correlation coefficient:", correlation)
        metrics = {
            'weighted_aucpr': aucpr,
            'fpr_tpr95': round(fpr_tpr95, 3),
            'fpr_tpr90': round(fpr_tpr90, 3),
            'fpr_tpr85': round(fpr_tpr85, 3),
            'fpr_tpr80': round(fpr_tpr80, 3),
            'fpr_tpr75': round(fpr_tpr75, 3)}
            # 'spearmanr':correlation
        # print(metrics)
        return metrics

    def calculate_metrics_perpixAP(self, labels, uncertainties):
        # if already exists, return
        if not self.overwrite and glob.glob(self.metric_save_path) and glob.glob(self.pr_save_path):
            print('result already generated at', self.metric_save_path)
            metrics = torch.load(self.metric_save_path)
            return metrics

        # concatenate lists for labels and uncertainties together
        if (labels[0].shape[-1] > 1 and np.ndim(labels[0]) > 2) or \
                (labels[0].shape[-1] == 1 and np.ndim(labels[0]) > 3):
            # data is already in batches
            labels = np.concatenate(labels)
            uncertainties = np.concatenate(uncertainties)
        else:
            labels = np.stack(labels)
            uncertainties = np.stack(uncertainties)
        labels = labels.squeeze()
        uncertainties = uncertainties.squeeze()

        # NOW CALCULATE METRICS
        pos = labels == 1 # ood, positive
        valid = np.logical_or(labels == 1, labels == 0)  # filter out void
        gt = pos[valid]
        del pos
        uncertainty = uncertainties[valid].reshape(-1).astype(np.float32, copy=False)
        del valid

        # Sort the classifier scores (uncertainties)
        sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
        uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
        # print('uncertainty', uncertainty)
        # print('gt', sum(gt[:3957]))
        del sorted_indices

        # Remove duplicates along the curve: remove the ones with the same unc
        distinct_value_indices = np.where(np.diff(uncertainty))[0] # get indices with that unc
        threshold_idxs = np.r_[distinct_value_indices, gt.size - 1] # add gt.size - 1 as the last element of distinct_value_indices
        print('unc', uncertainty[threshold_idxs][::-1][:3])
        print('total len', len(gt))
        # print('threshold_idxs', threshold_idxs)
        del distinct_value_indices, uncertainty
        
        # Accumulate TPs and FPs
        tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs] # num of positives at the diff unc levels
        print('tps', tps[::-1][:3])
        fps = 1 + threshold_idxs - tps
        print('fps', fps[::-1][:3])     
        del threshold_idxs

        # Compute Precision and Recall
        precision = tps / (tps + fps)
        # precision must be decreasing: flip, take cum max, flip
        precision = np.flip(np.maximum.accumulate(np.flip(precision, axis = 0)), axis = 0) #precision.flip(0).cummax(0)[0].flip(0)
        precision[np.isnan(precision)] = 0
        
        recall = tps / tps[-1]
        print('total', tps[-1])
        print('precision', precision[::-1][:5])
        print('recall', recall[::-1][:5])
        # stop when full recall attained and reverse the outputs so recall is decreasing
        sl = slice(tps.searchsorted(tps[-1]), None, -1)
        precision = np.r_[precision[sl], 1]
        recall = np.r_[recall[sl], 0]
        average_precision = -np.sum(np.diff(recall) * precision[:-1])
        aucpr = -round(np.trapz(precision, recall),5)

        pr_ts = {
            'precision': torch.from_numpy(precision),
            'recall': torch.from_numpy(recall),
            'tps': tps,
            'fps': fps,
            'aucpr': aucpr
            }
        if not self.debug: torch.save(pr_ts, self.pr_save_path)
        del pr_ts

        del precision, recall

        if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
            # Add an extra threshold position if necessary
            # to make sure that the curve starts at (0, 0)
            tps = np.r_[0., tps]
            fps = np.r_[0., fps]

        # Compute TPR and FPR
        tpr = tps / tps[-1]
        del tps
        fpr = fps / fps[-1]
        del fps

        # Compute AUROC, fpr_tpr95
        auroc = np.trapz(tpr, fpr)
        fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
        fpr_tpr90 = fpr[np.searchsorted(tpr, 0.90)]
        fpr_tpr85 = fpr[np.searchsorted(tpr, 0.85)]
        fpr_tpr80 = fpr[np.searchsorted(tpr, 0.80)]
        fpr_tpr75 = fpr[np.searchsorted(tpr, 0.75)]
        print('AP:', average_precision)
        print('FPR@95%TPR:', fpr_tpr95)
        # del curve_precision, curve_recall
        
        metrics = {
            'aucpr': aucpr,
            'auroc': auroc,
            'AP': round(average_precision, 3),
            'fpr_tpr95': round(fpr_tpr95, 3),
            'fpr_tpr90': round(fpr_tpr90, 3),
            'fpr_tpr85': round(fpr_tpr85, 3),
            'fpr_tpr80': round(fpr_tpr80, 3),
            'fpr_tpr75': round(fpr_tpr75, 3),

        }
        return metrics

    

    def get_unc_mask(self, rel_ood_img_path, get_label = False, get_freq = False, freq_type = 'image'):
        """get the unc tensor, mask (and freqeuncies)

        Args:
            rel_ood_img_path (list): _description_
            get_freq (bool, optional): if True, also return the frequencies. Defaults to False.
            pixel (bool, optional): if True, return the pixel wise freq, otherwise return the image wise one. Defaults to True.

        Returns:
            _type_: _description_
        """
        #get unc
        subname = extract_substring(rel_ood_img_path)
        if not self.baseline:
            unc_save_path = os.path.join(self.save_path, 'unc_results', self.cond_type, f'{subname}_{self.upsample_model}.pt')
            dic = torch.load(unc_save_path)
            unc_ts = dic[self.unc_type].detach().cpu()
            if len(unc_ts.shape) ==4: unc_ts = unc_ts[0] 
        else:
            unc_save_path = os.path.join(self.save_path, self.upsample_model, f'{subname}.pt')
            # unc_ts = torch.load(unc_save_path)
            dic = torch.load(unc_save_path)
            # print(dic)
            # print(self.unc_type)
            unc_ts = dic[self.unc_type].detach().cpu()
            unc_ts = F.interpolate(unc_ts.unsqueeze(0), self.image_size, mode='bilinear', align_corners=False)[0][0] #.detach().cpu()
        
        unc = unc_ts.numpy()
        # unc = unc[::self.portion, ::self.portion]
        if self.sqr: unc = unc**2 #square the unc
        '''no need for enlarging the image for the new sqare method'''
        # ts_max = unc_ts.max().item()
        # unc_ts = unc_ts/ts_max 
        # # change back to original shape
        # unc_pil = T.ToPILImage('L')(unc_ts)
        # transform = T.Compose([
        #             T.Resize((550, 688)),
        #             T.CenterCrop((550, 688)),
        #             T.ToTensor(),
        #         ])
        # unc = transform(unc_pil)[0].numpy() * ts_max
        if self.do_filter: unc[unc < self.threshold] = 0
        if get_freq:
            label_path = os.path.join(self.label_save_dir, rel_ood_img_path[:-4])+ '.pt'
            dic = torch.load(label_path)
            freq = dic[freq_type][::self.portion, ::self.portion]

            if not get_label:
                return unc, freq.detach().cpu().numpy()
            else:
                if self.rellis:
                    label_ar = dic['label'] #.numpy() need this for rugd
                else:
                    label_ar = dic['label'].numpy() 
                # label_ar = label_ar[::self.portion, ::self.portion]
                return unc, freq.detach().cpu().numpy(), label_ar
        else:
            # get mask
            mask_path = os.path.join(self.mask_parent_path, f'{rel_ood_img_path[:-4]}.png')
            im = Image.open(mask_path)
            mask_bool = np.array(im) # dtype=bool ->  size=(550, 688)
            mask = mask_bool.astype(int) 
            return unc, mask, mask_bool
       

    def metric_eval(self):
        total_metrics = dict()
        if self.auprc:
            p_metrics = self.pixel_metric()
            total_metrics = total_metrics | p_metrics
        if self.weighted_auprc:
            for freq_type in self.freq_types:
                p_metrics = self.weighted_pixel_metric(freq_type)
                total_metrics = total_metrics | p_metrics
        if self.weighted_obj:
            for freq_type in self.freq_types:
                print(freq_type)
                o_best_metric, o_total_metrics = self.weighted_obj_metric(freq_type)
                total_metrics = total_metrics | o_best_metric
        if self.obj:
            o_best_metric, o_total_metrics = self.obj_metric()
            total_metrics = total_metrics | o_best_metric
        if self.psnr:
            psnr_metric = self.psnr_metric()
            total_metrics = total_metrics | psnr_metric
        return total_metrics
        
    def psnr_metric(self):
        print('start psnr')
        ood_score_sum = 0
        ind_score_sum = 0
        max_ood_score = 0

        for rel_ood_img_path in self.image_paths:
            unc, mask, mask_bool = self.get_unc_mask(rel_ood_img_path)
            # make the void pixel has values 1-2 = -1

            # ood and ind score
            ood_score_sum += unc[mask_bool].mean()
            ind_score_sum += unc[~mask_bool].mean()
            max_ood_score = max([max(unc[mask_bool]), max_ood_score])
        avg_ood_score = round(ood_score_sum/len(self.image_paths), 3)
        avg_ind_score = round(ind_score_sum/len(self.image_paths), 3)
        psnr = 20 * np.log10(max_ood_score/avg_ind_score)

        metrics = {'PSNR': psnr, 'avg_ood_score': avg_ood_score, 'avg_ind_score': avg_ind_score, 'max_ood_score': max_ood_score}
        return metrics

    
    def weighted_pixel_metric(self, freq_type):
        # freq_type = 'image'
        if self.rellis:
            auprc_norm = {'image': 0.68175, 'pixel': 1, 'new_pixel': 1, 'new_image': 1}
        else:
            auprc_norm = {'pixel': 0.92652, 'image':0.87487}
        # weighted: use the weighted auprc
        weighted_metric_save_path = os.path.join(self.result_save_path, 'weighted_metrics', f'{self.cond_type}_{self.upsample_model}_{self.unc_type}_{freq_type}.pt')
        weighted_pr_save_path = os.path.join(self.result_save_path, 'weighted_pr_results', f'{self.cond_type}_{self.upsample_model}_{self.unc_type}_{freq_type}.pt')
        if not self.overwrite and glob.glob(weighted_metric_save_path):
            metrics = torch.load(weighted_metric_save_path)
            print('already generated at', weighted_metric_save_path)
            return metrics
        
        uncertainties = []
        labels = []
        c = 0
        for rel_ood_img_path in self.image_paths:
            if c%100 == 0:  
                print('progress:', c, rel_ood_img_path)
            c+=1
            unc, freq  = self.get_unc_mask(rel_ood_img_path, get_freq=True, freq_type = freq_type)
            # make the void pixel has values 1-2 = -1
            if freq_type == 'pixel': 
                freq[freq == 1.65e-4] = 2
                freq = np.where(freq == 2, freq, freq / 0.433998)
                # freq = freq /0.433998
            elif freq_type == 'new_pixel': 
                freq = np.where(freq == 2, freq, freq / 0.330252)
            else: # image
                freq[freq ==  0.198142] = 2
            weight = 1 - freq
            uncertainties.append(unc)
            labels.append(weight)
         
    
        # plot and calculate aucpr and other metrics
        # metrics = self.calculate_weighted_aucpr(labels, labels, weighted_pr_save_path, weighted_metric_save_path)
        metrics = self.calculate_weighted_aucpr(labels, uncertainties, weighted_pr_save_path, weighted_metric_save_path)
        # print(metrics)
        metrics = {f'weighted_aucpr_{freq_type}_norm': metrics['weighted_aucpr']/auprc_norm[freq_type]} | {f'{key}_{freq_type}': val for key, val in metrics.items()}
        print(metrics)
                #    f'spearmanr_{freq_type}': metrics['spearmanr']}
        
        if not self.debug: torch.save(metrics, weighted_metric_save_path)
        return metrics
    
    def pixel_metric(self):
        # weighted: use the weighted auprc
        if not self.overwrite and glob.glob(self.metric_save_path):
            metrics = torch.load(self.metric_save_path)
            print('already generated at', self.metric_save_path)
            return metrics
        uncertainties = []
        labels = []
        
        ood_score_sum = 0
        ind_score_sum = 0
        
        for rel_ood_img_path in self.image_paths:
            unc, mask, mask_bool = self.get_unc_mask(rel_ood_img_path)
        
            if self.auprc:
                uncertainties.append(unc)
                labels.append(mask)

            # ood and ind score
            ood_score_sum += unc[mask_bool].mean()
            ind_score_sum += unc[~mask_bool].mean()
            
        avg_ood_score = round(ood_score_sum/len(self.image_paths), 3)
        avg_ind_score = round(ind_score_sum/len(self.image_paths), 3)
        # norm_avg_ood_score = round(avg_ood_score/ts_max, 3)
        # norm_avg_ind_score = round(avg_ind_score/ts_max, 3)
        
        # plot and calculate aucpr and other metrics
        metrics = self.calculate_metrics_perpixAP(labels, uncertainties)

        print('avg score on ood:', avg_ood_score)
        print('avg score on ind:', avg_ind_score)
        metrics = metrics | {'avg_ood_score': avg_ood_score, 'avg_ind_score': avg_ind_score}
        print(metrics)
        if not self.debug: torch.save(metrics, self.metric_save_path)
        return metrics

    def get_max_unc(self):
        ts_max = 0
        for rel_ood_img_path in self.image_paths:
            unc, mask, mask_bool = self.get_unc_mask(rel_ood_img_path)
            curr_ts_max = unc.max().item()
            if ts_max < curr_ts_max:
                ts_max = curr_ts_max
        return ts_max

    def weighted_obj_metric(self, freq_type):
        metric_save_path = os.path.join(self.result_save_path, f'weighted_{freq_type}_obj_metrics', f'{self.cond_type}_{self.upsample_model}_{self.unc_type}.pt') #self.weighted_pixel_obj_metric_save_path
        print('weighted_obj_metric save path', metric_save_path)
        key_label = freq_type
        if not self.overwrite and glob.glob(metric_save_path):
            print('metric already exists at', metric_save_path)
            total_metrics = torch.load(metric_save_path)
            best_metric = {key: total_metrics[key] for key in [f'f1*_{key_label}', f'TP_{key_label}', f'FP_{key_label}', f'FN_{key_label}', f'TN_{key_label}']}
            return best_metric, total_metrics

        total_metrics = dict()
        core_metric_list = [] #f1, sIoU_gt, prec_pred
        norm_factor = self.get_max_unc()
        print('norm_factor', norm_factor)


         
        [TPs, FPs,  FNs, TNs] = [np.zeros(len(self.obj_thresh_p)) for i in range(4)]
        c = 0
        for rel_ood_img_path in self.image_paths:
            if c%500 == 0:  
                print('progress:', c, rel_ood_img_path)
            c+=1
            unc, freq, label = self.get_unc_mask(rel_ood_img_path, get_label= True, get_freq=True, freq_type=freq_type)
            unc /= norm_factor
            con_label = relabel_connected_components(label, threshold = self.obj_thresh_segsize)
            if freq_type == 'pixel': freq = np.where(freq == 2, freq, freq / 0.433998) #freq = freq /0.433998
            if freq_type == 'new_pixel': freq = np.where(freq == 2, freq, freq / 0.330252) #freq = freq /0.433998
            weight = 1 - freq
            TP, FP, FN, TN = calculate_weighted(con_label, unc, weight, self.obj_thresh_p)

            TPs += TP
            FPs += FP
            FNs += FN
            TNs += TN
                
        f1s = (2 * TPs) / (2 * TPs + FNs + FPs)
        f1, maxid= f1s.max(), f1s.argmax()
        best_metric = {f'f1*_{key_label}': f1, f'TP_{key_label}': TPs[maxid], f'FP_{key_label}': FPs[maxid], f'FN_{key_label}': FNs[maxid], f'TN_{key_label}': TNs[maxid]}
        total_metrics = best_metric | {f'f1s_{key_label}': f1s, f'TPs_{key_label}': TPs, f'FPs_{key_label}': FPs, f'FNs_{key_label}': FNs, f'TNs_{key_label}': TNs}
        if not self.debug: torch.save(total_metrics, metric_save_path)
        print('unc thresholds:', self.obj_thresh_p)
        print(total_metrics)
        return best_metric, total_metrics
    
    def obj_metric(self):
        if not self.overwrite and glob.glob(self.obj_metric_save_path):
            print('metric already exists at', self.obj_metric_save_path)
            total_metrics = torch.load(self.obj_metric_save_path)
            # print(total_metrics)
            core_metric_list = []
            for all_result in total_metrics.values():
                core_metric_list.append({'f1_mean': all_result.f1_mean, 'sIoU_gt': all_result.sIoU_gt, 'prec_pred': all_result.prec_pred})
            best_metric = max(core_metric_list, key=lambda x: x['f1_mean'])
            return best_metric, total_metrics

        total_metrics = dict()
        core_metric_list = [] #f1, sIoU_gt, prec_pred
        norm_factor = self.get_max_unc()
        print('norm_factor', norm_factor)
        for thresh_p in self.obj_thresh_p:
            thresh_p = round(thresh_p, 3) # round to avoid values like 0.399999
            print(thresh_p)
            obj_result_list = []
            for rel_ood_img_path in self.image_paths:
                unc, mask, mask_bool = self.get_unc_mask(rel_ood_img_path)
                unc /= norm_factor
                anomaly_gt, anomaly_pred, mask = default_instancer(unc, mask, thresh_p = thresh_p, thresh_segsize = self.obj_thresh_segsize, 
                                                                   thresh_instsize = self.obj_thresh_segsize) # also set component size threshold for gt mask
                result = segment_metrics(anomaly_gt, anomaly_pred, self.iou_thresholds)
                obj_result_list.append(result)
                # if result.sIoU_gt.mean()<0.1: print(rel_ood_img_path, result.sIoU_gt)
                
            
            all_result = aggregate(thresh_sIoU = self.thresh_sIoU, frame_results= obj_result_list, method_name = 'trial', dataset_name = 'rugd', verbose = self.obj_verbose)
            total_metrics[thresh_p] = all_result
            core_metric_list.append({'f1_mean': all_result.f1_mean, 'sIoU_gt': all_result.sIoU_gt, 'prec_pred': all_result.prec_pred})
        if not self.debug: torch.save(total_metrics, self.obj_metric_save_path)
        best_metric = max(core_metric_list, key=lambda x: x['f1_mean']) #f1_star, sIoU_gt, prec_pred

        return best_metric, total_metrics


def get_curve_pr(precision, recall, num_points = 50):
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size()[0] - 1
    interval = 1.0 / num_points
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall
    return curve_precision, curve_recall

def plot_pr(cond_type, upsample_model, result_save_path, unc_types = ['LowRes', 'HighRes', 'SAM_Pixel', 'SAM_LowRes', 'SAM_HighRes', 
                     'LowRes_cos', 'HighRes_cos', 'SAM_Pixel_cos', 'SAM_LowRes_cos', 'SAM_HighRes_cos'], num_points = 200, 
                     label = ''):
    fig, ax = plt.subplots()
    for unc_type in unc_types: 
            pr_save_path = os.path.join(result_save_path, 'weighted_pr_results', f'{cond_type}_{upsample_model}_{unc_type}_{label}.pt')
            pr_ts = torch.load(pr_save_path)
            precision, recall = pr_ts['precision'], pr_ts['recall']
            curve_precision, curve_recall = get_curve_pr(precision, recall, num_points = num_points)
            # print(curve_precision, curve_recall)
            if 'aucpr' in pr_ts:
                aucpr = pr_ts['aucpr']
            else:
                aucpr = -round(np.trapz(precision, recall),3)
                pr_ts['aucpr'] = aucpr
                torch.save(pr_ts, pr_save_path)

            del precision, recall
            # plt.plot(curve_recall, curve_precision,'.', label=f'{unc_type} aucpr = {aucpr}')
            plt.plot(curve_recall, curve_precision,'.', label=f'{unc_type} {label} aucpr = {aucpr}')
            del curve_recall, curve_precision
            
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(f'{cond_type} {upsample_model}')
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    plot_save_path = os.path.join(result_save_path, 'PRcurve', f'{cond_type}_{upsample_model}_{label}.png')
    plt.savefig(plot_save_path, bbox_inches='tight', bbox_extra_artists=[legend]) #, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.show()
    plt.clf()


if __name__ == '__main__':
    pass