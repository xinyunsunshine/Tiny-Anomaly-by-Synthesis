import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import os, glob
import numpy as np
from diffunc.all import load_input_img,  get_image_paths, extract_substring
from PIL import Image

test_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood/test'

diffunc_save_path = '/home/sunsh16e/diffunc/experiment/rugd_sqr_new' #unc path
base_save_path = '/home/sunsh16e/diffunc/baseline/data/knn/unc' #unc path
base_save_path_gmm = '/home/sunsh16e/diffunc/baseline/data/gmm/unc' #unc path
gen_save_path_gmm = '/home/sunsh16e/diffunc/baseline/data/gmm/gen_unc' #unc path for gen gmm
# result_save_path = '/home/sunsh16e/diffunc/baseline/data/eval' # eval result save path
mask_parent_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/masks/ood/test'
mask_parent_path_train = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/masks/ood/train'
fig_save_dir = '/home/sunsh16e/diffunc/baseline/data/compare_exp'
upsample_model = 'dinov2'
cond_type = 'exp_linear_p'
unc_type = 'SAM_HighRes_cos'

all_folders = ['park-8'] #['trail-13', 'park-8', 'trail-5', 'trail-7', 'creek', 'park-1']
for folder in all_folders:
    image_path_name = sorted(get_image_paths(test_dataset_path, folders = [folder]))[::5]
    print(folder, 'len', len(image_path_name))
    print(image_path_name[:5])

    column_num = 4
    row_num = 3
    gmm_max=[]
    knn_max=[]
    for rel_ood_img_path in image_path_name:
        #### mask
        mask_path = os.path.join(mask_parent_path, rel_ood_img_path)
        if not glob.glob(mask_path):
            mask_path = os.path.join(mask_parent_path_train, rel_ood_img_path)
        im = Image.open(mask_path)
        mask_bool = np.array(im) # dtype=bool ->  size=(550, 688)
        mask = mask_bool.astype(int) 

        #### diffunc
        subname = extract_substring(rel_ood_img_path)
        # basics    
        basics_save_path = os.path.join(diffunc_save_path, 'unc_results', cond_type, f'{subname}_basics.pt')
        base_dic = torch.load(basics_save_path)
        ood_img, ind_img, seg_ts = base_dic['ood'], base_dic['ind'], base_dic['seg']
        original, generated, seg_ts = ood_img.detach().cpu(), ind_img.detach().cpu(), seg_ts.detach().cpu()
        # unc
        unc_save_path = os.path.join(diffunc_save_path, 'unc_results', cond_type, f'{subname}_{upsample_model}.pt')
        dic = torch.load(unc_save_path)
        unc_ts = dic['SAM_HighRes_cos'].detach().cpu()
        unc_diffunc = unc_ts.numpy()


        ###### baselines
        unc_save_path = os.path.join(base_save_path, upsample_model, f'{subname}.pt')
        unc_ts = torch.load(unc_save_path)['no_sam']
        unc_ts = F.interpolate(unc_ts.unsqueeze(0), (550, 688), mode='bilinear', align_corners=False)[0][0].detach().cpu()
        unc_base = unc_ts.numpy()
        knn_max.append(unc_base.max())

        unc_save_path = os.path.join(base_save_path, upsample_model, f'{subname}.pt')
        unc_ts_sam = torch.load(unc_save_path)['sam']
        unc_ts_sam = F.interpolate(unc_ts_sam.unsqueeze(0), (550, 688), mode='bilinear', align_corners=False)[0][0].detach().cpu()
        unc_base_sam = unc_ts_sam.numpy()

        unc_save_path = os.path.join(base_save_path_gmm, upsample_model, f'{subname}.pt')
        unc_ts = torch.load(unc_save_path)['no_sam']
        unc_ts = F.interpolate(unc_ts.unsqueeze(0), (550, 688), mode='bilinear', align_corners=False)[0][0].detach().cpu()
        unc_base_gmm = unc_ts.numpy()
        gmm_max.append(unc_base_gmm.max())

        unc_save_path = os.path.join(base_save_path_gmm, upsample_model, f'{subname}.pt')
        unc_ts_sam = torch.load(unc_save_path)['sam']
        unc_ts_sam = F.interpolate(unc_ts_sam.unsqueeze(0), (550, 688), mode='bilinear', align_corners=False)[0][0].detach().cpu()
        unc_base_sam_gmm = unc_ts_sam.numpy()

        fig = plt.figure(frameon=False, figsize=(3*column_num, 2.8 * row_num))

        plt.subplot(row_num, column_num, 1)
        plt.imshow(original.permute(1, 2, 0), cmap='gray')
        plt.title('original')
        plt.axis('off')

        plt.subplot(row_num, column_num, 2)
        plt.imshow(generated.permute(1, 2, 0), cmap='gray')
        plt.title('generated')
        plt.axis('off')

        plt.subplot(row_num, column_num, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('mask')
        plt.axis('off')

        plt.subplot(row_num, column_num, 4)
        plt.imshow(seg_ts.permute(1, 2, 0), cmap='gray')
        plt.title('sam')
        plt.axis('off')
        

        plt.subplot(row_num, column_num, column_num + 1)
        # unc_diffunc[unc_diffunc<0.5] =0
        plt.imshow(dic['HighRes_cos'].detach().cpu()) 
        plt.title('diffunc')
        plt.axis('off')

        plt.subplot(row_num, column_num, column_num + 2)
        # unc_diffunc[unc_diffunc<0.5] =0
        plt.imshow(dic['SAM_HighRes_cos'].detach().cpu()) 
        plt.title('diffunc SAM')
        plt.axis('off')

        plt.subplot(row_num, column_num, column_num + 3)
        # plt.imshow(mask, cmap='gray')
        # unc_base[unc_base<15] = 0
        plt.imshow(unc_base, alpha = 1) 
        plt.title('knn')
        plt.axis('off')

        plt.subplot(row_num, column_num, column_num + 4)
        # plt.imshow(mask, cmap='gray')
        # unc_base[unc_base<15] = 0
        plt.imshow(unc_base_sam, alpha = 1) 
        plt.title('knn SAM')
        plt.axis('off')

        plt.subplot(row_num, column_num, column_num*2 + 1)
        # plt.imshow(mask, cmap='gray')
        # unc_base[unc_base<15] = 0
        plt.imshow(unc_base_gmm, alpha = 1) 
        plt.title('gmm')
        plt.axis('off')

        plt.subplot(row_num, column_num, column_num*2 + 2)
        # plt.imshow(mask, cmap='gray')
        # unc_base[unc_base<15] = 0
        plt.imshow(unc_base_sam_gmm, alpha = 1) 
        plt.title('gmm SAM')
        plt.axis('off')

        plt.suptitle(subname)

        fig_save_path = os.path.join(fig_save_dir, folder, f'{subname}.png')
        fig.savefig(fig_save_path, dpi=300)