from eval import PReval, plot_pr
from diffunc.all import get_image_paths
import numpy as np
import pandas as pd
import os

# method
diffunc = False
table_save_label = 'no_transform' #'perfect_img_auprc'
# metrics used
freq_types = ['image']
weighted_obj = True
weighted_auprc = True
psnr = False
auprc = False
obj = False
sqr = False

# process
debug = False
overwrite = False
do_filter = False
threshold = 300

# obj  metric
obj_thresh_segsize = 500 #2000
obj_verbose = False
obj_thresh_p = np.linspace(0.2, 0.9, 8, endpoint=True) #for pixel only np.linspace(0, 0.1, 8, endpoint=True) #[0.2, 0.3, 0.7, 0.8] #np.linspace(0.4, 0.6, 3, endpoint=True) #17


mask_parent_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/masks/ood'
dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
label_save_dir = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/labels/ood'

image_paths = sorted(get_image_paths(dataset_path)) #['trail-7/trail-7_00011.png'] 

if diffunc:
    cond_types = ['l2_p'] #, 'exp_linear_p'] #, 'exp_linear_p'] #'l2_p', 'l2_q', 'exp_linear_q', 'exp_linear_p'
    unc_types = {'l2': ['SAM_HighRes_cos'], 'l2_p': ['SAM_HighRes_cos'], 'l2_q': ['SAM_HighRes_cos'], 'exp_linear_q': ['SAM_HighRes_cos'], 'exp_linear_p': ['SAM_HighRes_cos', 'HighRes_cos', 'SAM_HighRes_cos_sqr']}
                # ['LowRes', 'HighRes', 'SAM_LowRes', 'SAM_HighRes', 'SAM_HighRes_cos_sqr'
                #                     'LowRes_cos', 'HighRes_cos', 'SAM_LowRes_cos', 'SAM_HighRes_cos']
    save_path = '/home/sunsh16e/diffunc/experiment/rugd_sqr_new' #unc path
    result_save_path = '/home/sunsh16e/diffunc/experiment/rugd_sqr_new/eval' # eval result save path
else:
    cond_types =['gmm', 'knn'] # ['gmm', 'knn'] 
    unc_types = {'gmm': ['no_sam', 'sam'], 'knn': ['no_sam', 'sam']} # ['no_sam', 'sam']
    result_save_path = '/home/sunsh16e/diffunc/baseline/data/eval' # eval result save path



total = dict()
for cond_type in cond_types:
    if not diffunc: save_path = f'/home/sunsh16e/diffunc/baseline/data/{cond_type}/unc' # baseline unc path
    for upsample_model in ['resnet50']: #['dino16', 'dinov2', 'clip', 'maskclip', 'vit', 'resnet50']:
        for unc_type in unc_types[cond_type]:
            print(cond_type, upsample_model, unc_type)
            pr = PReval(image_paths, save_path, mask_parent_path, result_save_path, cond_type, upsample_model, unc_type, 
                        do_filter, threshold, auprc = auprc, overwrite = overwrite, debug = debug, sqr= sqr, label_save_dir = label_save_dir,
                        weighted_auprc= weighted_auprc, psnr = psnr, baseline = (not diffunc),
                        weighted_obj = weighted_obj,freq_types = freq_types,
                        obj = obj, obj_thresh_p = obj_thresh_p, obj_thresh_segsize = obj_thresh_segsize, obj_verbose = obj_verbose)

            metric = pr.metric_eval()
            metric['upsample_model'] = upsample_model
            metric['unc_type'] = unc_type
            total[f'{cond_type}_{upsample_model}_{unc_type}'] = metric

#         # plot_pr(cond_type, upsample_model, result_save_path, unc_types = unc_types, label = metric_type)
print(total)
if not debug:
    df = pd.DataFrame.from_dict(total, orient='index')
    csv_path = os.path.join(result_save_path, 'tables', f'{table_save_label}.csv')
    df.to_csv(csv_path, index=True)

# # upsample_model = 'dino16'
# # cond_type = l2_q'
# # plot_pr(cond_type, upsample_model)
# # df = pd.DataFrame.from_dict(total, orient='index')
# # df.to_csv(f'{cond_type}.csv', index=True)

''''change sam unc'''
# from diffunc.sam import Segmentation, show_image_and_masks
# from diffunc.all import load_sam
# dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
# image_paths = sorted(get_image_paths(dataset_path))
# mask_generator = load_sam()
# unc_types = ['LowRes', 'HighRes', 'LowRes_cos', 'HighRes_cos']
# count = 0
# cond_type = 'l2_q' #'l2_q', 'exp_linear_q', 'exp_linear_p' 
# for rel_ood_img_path in image_paths:
#     ori_img_path = os.path.join(dataset_path, rel_ood_img_path)
#     ood_img_pil = Image.open(ori_img_path).convert("RGB")
#     seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3, verbose = False)
#     short_img_path = extract_substring(rel_ood_img_path)

#     for upsample_model in ['dino16', 'dinov2', 'clip', 'maskclip', 'vit', 'resnet50']:
#         unc_path = f'/home/sunsh16e/diffunc/experiment/rugd_sqr/unc_results/{cond_type}/{short_img_path}_{upsample_model}.pt'
#         uc = torch.load(unc_path) 
#         for unc_type in unc_types:
#             uc['SAM_'+ unc_type] = seg.regularize(uc[unc_type], fill=0)
#         torch.save(uc, unc_path)
#     if count%10 == 0: print(count, rel_ood_img_path)
#     count+=1