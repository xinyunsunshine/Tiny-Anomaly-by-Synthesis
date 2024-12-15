from diffunc.all import load_input_img,  get_image_paths, extract_substring, load_sam
import os
import glob
import torch
from PIL import Image
from diffunc.sam import Segmentation

base = False

baseline = 'gmm' #'knn' #'gmm'

way = False #wayfarer
if way:
    test_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood/test'
else:
    test_dataset_path = '/home/gridsan/groups/rrg/data/RUGD/ddpm_train_sets/full/images/ood/'
train_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/id/train'
diffunc_save_path = '/home/gridsan/sjiang/diffunc/experiment/rugd_sqr_new'
save_path = '/home/sunsh16e/diffunc/baseline/data'
parent_dir = '/home/sunsh16e/diffunc/baseline/data'

mask_generator = load_sam()

train_bank_size = 50000
upsample_models = ['dinov2'] # ['dino16', 'clip', 'maskclip', 'vit', 'resnet50'] 
cond_type = ['exp_linear_p'] #, 'exp_linear_p']
print(cond_type)
highres = True
n_components = 20 #200
train_max_iter=50
train_n_init = 1

test_image_paths = sorted(get_image_paths(test_dataset_path))
print(len(test_image_paths))

for upsample_model in upsample_models[::-1]:

    count = 0
    for rel_ood_img_path in test_image_paths:
    for rel_ood_img_path in test_image_paths:
        if count%10 ==0:
            print(count, rel_ood_img_path)
        count+=1

        ori_img_path = os.path.join(test_dataset_path, rel_ood_img_path)
        ood_img_pil = Image.open(ori_img_path).convert("RGB")
        seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3, verbose = False)

        if base: # baseline
            unc_save_path = os.path.join(parent_dir,  baseline, 'unc', upsample_model, f'{extract_substring(rel_ood_img_path)}.pt')
            unc_ts = torch.load(unc_save_path)
            if type(unc_ts) == dict: # already did sam
                continue
            unc_sam = seg.regularize(unc_ts, fill=0)
            new_ts = {'no_sam': unc_ts, 'sam': unc_sam }
            torch.save(new_ts, unc_save_path)
        else: # diffunc
            unc_save_path = os.path.join(diffunc_save_path, 'unc_results', cond_type, f'{extract_substring(rel_ood_img_path)}_{upsample_model}.pt')
            unc_dic = torch.load(unc_save_path)
            unc_ts = unc_dic['HighRes_cos']**2
            unc_sam = seg.regularize(unc_ts, fill=0)
            new_ts = unc_dic | {'SAM_HighRes_cos_sqr': unc_sam, 'HighRes_cos_sqr': unc_ts, 'SAM_HighRes_cos_sqr_simp': unc_dic['SAM_HighRes_cos']**2}
            torch.save(new_ts, unc_save_path)
        

        