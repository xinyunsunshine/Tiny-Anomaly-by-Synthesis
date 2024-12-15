# 2024.4.26 Sunshine Jiang
# code to run the whole pipeline on all images and store the unc heatmaps

# check if the corresponding result exits
# if not, run the pipeline and store
# visaulize the result for every n images

# params: dataset path, conditioners to use, upsample methods to use, save path
import time
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np

from IPython.display import Image as IPyImage
from IPython.display import display
import jupyviz as jviz
import matplotlib.pyplot as plt

from featup.util import norm
from square import SquareTiledFeatUpPipeline

import os
import glob

from ssim_diffunc import calculate_ssim
from denoising_diffusion_pytorch import Unet
from diffunc import viz
from diffunc.sam import Segmentation, show_image_and_masks
from diffunc.all import (
    extract_substring,
    GaussianDiffusion,
    ScheduledParams,
    Conditioner,
    L2Objective,
    ExponentialPriorSPMObjective,
    set_random_seed,
    load_input_img,
    download_from_cloud,
    softrect_inlier,
    get_image_paths,
    load_sam
)

TIME = False
def get_pipe(img_ts, upsampler):
    img_pil = T.ToPILImage()(img_ts)
    pipeline = SquareTiledFeatUpPipeline(upsampler, img_pil)
    return pipeline

def get_total_unc(diff_func, pipe_ori_feats, pipr_gen_feats, pipe, upsample):
    ncertainty_list = []
    for ood_hr_feats, ind_hr_feats in zip(pipe_ori_feats, pipr_gen_feats):
        diff_hr_cos  = diff_func(ood_hr_feats, ind_hr_feats).unsqueeze(0)
        ncertainty_list.append(diff_hr_cos)
    uncertainty_per_square_image = torch.stack(ncertainty_list, dim=0)
    final_unc = pipe.reassemble_partitions(uncertainty_per_square_image).squeeze(0)
    return upsample(final_unc.unsqueeze(0).unsqueeze(1))[0][0]



@torch.no_grad()
def experiment(ood_img, ind_img, upsampler, seg, upsample, pixel_up, size=None, square = True):
    featup_start = time.time()
    if square:
        diff_l2 = lambda x, y: (x - y).square().sum(dim=0)  # (1, 3, H, W) -> (1, H, W)
        diff_cos = lambda x, y: 1-F.cosine_similarity(x,y, dim = 0) # (1, 3, H, W) -> (1, H, W)
    
        diff_pixel = upsample(diff_l2(ood_img, pixel_up(ind_img)).unsqueeze(0).unsqueeze(0))[0][0].cuda()
        diff_pixel_cos  = upsample(diff_cos(ood_img, pixel_up(ind_img)).unsqueeze(0).unsqueeze(0))[0][0].cuda()

        pipe_ori = get_pipe(ood_img, upsampler)
        pipr_gen = get_pipe(ind_img, upsampler)
        # print('high res features len', len(pipe_ori.hi_res_feats))
        diff_hr = get_total_unc(diff_l2, pipe_ori.hi_res_feats, pipr_gen.hi_res_feats, pipe_ori, upsample)
        diff_lr = get_total_unc(diff_l2, pipe_ori.lo_res_feats, pipr_gen.lo_res_feats, pipe_ori, upsample)
        diff_hr_cos = get_total_unc(diff_cos, pipe_ori.hi_res_feats, pipr_gen.hi_res_feats, pipe_ori, upsample)
        diff_lr_cos = get_total_unc(diff_cos, pipe_ori.lo_res_feats, pipr_gen.lo_res_feats, pipe_ori, upsample)

        
        
    else:
        diff_l2 = lambda x, y: (x - y).square().sum(dim=1)  # (1, 3, H, W) -> (1, H, W)
        diff_cos = lambda x, y: 1-F.cosine_similarity(x,y) # (1, 3, H, W) -> (1, H, W)
        maybe_resize = T.Resize(size) if size is not None else nn.Identity()
        ood_input = maybe_resize(ood_img)  # (1, 3, H, W)
        ind_input = maybe_resize(ind_img)  # (1, 3, H, W)

        # ood feats
        ood_hr_feats = upsampler(norm(ood_input))
        ood_lr_feats = upsampler.model(norm(ood_input))

        # ind feats
        ind_hr_feats = upsampler(norm(ind_input))
        ind_lr_feats = upsampler.model(norm(ind_input))

        # compute L2 distances
        diff_pixel = diff_l2(ood_input, ind_input)  # (1, H, W)
        diff_lr = diff_l2(ood_lr_feats, ind_lr_feats)  # (1, H, W)
        diff_hr = diff_l2(ood_hr_feats, ind_hr_feats)  # (1, H, W)
        diff_pixel_cos  = diff_cos(ood_input, ind_input)  # (1, H, W)
        diff_lr_cos  = diff_cos(ood_lr_feats, ind_lr_feats)  # (1, H, W)
        diff_hr_cos  = diff_cos(ood_hr_feats, ind_hr_feats)  # (1, H, W)
    if TIME:
        featup_finish = time.time()  # End of process 1
        featup_time = featup_finish - featup_start
        print(f"Time for featup: {featup_time:.2f} seconds")
    samed_unc = [seg.regularize(e, fill=0) for e in [diff_pixel, diff_lr, diff_hr, diff_pixel_cos, diff_lr_cos, diff_hr_cos, diff_hr_cos**2]]
    if TIME:
        sam_finish = time.time()  # End of process 1
        print(f"Time for sam: {sam_finish - featup_finish:.2f} seconds")
    return [diff_lr, diff_hr, diff_lr_cos, diff_hr_cos, diff_hr_cos**2] + samed_unc

def plot_all_diff(ood_img, ind_img, seg_ts, num_rows):
    ### need to DIVIDE BY 255 here to restore the same effect as in in prev notebooks since not normalized here
    if len(ood_img.shape)==3: # sqr case
        original, generated = ood_img.detach().cpu(), ind_img.detach().cpu()
    else:
        original, generated = ood_img[0].detach().cpu(), ind_img[0].detach().cpu()

    # print(original.shape, generated.shape)
    diff=(original/255-generated/255).abs().float()  # (1, 3, H, W)
    diff_norm: Tensor = diff.norm(dim=0)
    diff_prod: Tensor = diff.prod(dim=0)  # (1, H, W)
    diff_softrect: Tensor = 1 - softrect_inlier(original/255, generated/255, width=0.1, steepness=100).prod(dim=0)  # (1, H, W)
    
    # this line moved to unc_pipelien func: fig = plt.figure(figsize=(16, 3*num_rows))
    ax1 = plt.subplot(num_rows, 5, 1)
    plt.imshow(original.permute(1, 2, 0))
    ax1.set_title('original')
    
    ax2 = plt.subplot(num_rows, 5, 2)
    plt.imshow(generated.permute(1, 2, 0))
    ax2.set_title('generated')
    
    ax3 = plt.subplot(num_rows, 5, 3)
    plt.imshow(diff_norm)
    ax3.set_title('norm diff')
    
    ax3 = plt.subplot(num_rows, 5, 4)
    plt.imshow(diff_prod)
    ax3.set_title('prod diff')
    
    ax3 = plt.subplot(num_rows, 5, 5)
    plt.imshow(seg_ts.detach().cpu().permute(1, 2, 0))
    ax3.set_title('SAM w/ 6 masks')

def diff_subplot(row, num_rows, upsample_model, diff_dic, 
                square = True,
                 title_dic = {'LowRes': 'L2 Low-Res Feats', 
                 'HighRes': 'L2 High-Res Feats',
                 'SAM_Pixel': 'SAM L2 Pixel-space', 
                 'SAM_LowRes': 'SAM L2 Low-Res Feats', 
                 'SAM_HighRes': 'SAM L2 High-Res Feats'}):
  
    idx = 0
    for key in title_dic:
        diff = diff_dic[key]
        ax1 = plt.subplot(num_rows, 5, (row-1)*5+idx+1)
        if square:
            plt.imshow(diff.detach().cpu())
        else:
            plt.imshow(diff[0].detach().cpu())

        ax1.set_title(title_dic[key])
        if idx==0:
            ax1.set_ylabel(upsample_model)
        idx += 1

    # for key, diff in diff_dic.items():
    #     # if (row-1)*5+idx+1>25:
    #     #     print('row, idx', row, idx)
    #     ax1 = plt.subplot(num_rows, 5, (row-1)*5+idx+1)
    #     plt.imshow(diff[0].detach().cpu())
    #     ax1.set_title(title_dic[key])
    #     if idx==0:
    #         ax1.set_ylabel(upsample_model)
    #     idx += 1

class OneImgPipe():
    def __init__(self, dataset_path, save_path,rel_ood_img_path, image_size, cond_type, conditioner, upsampler, upsampler_name, ori_image_size,
                 verbose = False, only_generate_img = False, square = True, gen_save_path = None, ssim = False, 
                 ddim = False, ddim_sampling_timesteps = 250, ddim_guidance_scale = 0.5):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.rel_ood_img_path = rel_ood_img_path
        self.image_size  = image_size
        self.ori_image_size = ori_image_size
        self.cond_type = cond_type
        self.conditioner = conditioner
        self.upsampler = upsampler
        self.upsampler_name = upsampler_name
        self.verbose = verbose
        self.only_generate_img = only_generate_img #only generate diffusion images, but not get unc map
        self.square = square # do the cutting into square and merging for better alignment
        self.gen_save_path = gen_save_path
        self.ssim = ssim
        self.ddim = ddim
        self.ddim_sampling_timesteps = ddim_sampling_timesteps
        self.ddim_guidance_scale = ddim_guidance_scale

        ###### related paths ######
        # diffusion generated img path
        self.pure_img_path = extract_substring(rel_ood_img_path, keep_ext = True)
        self.gen_img_path = os.path.join(self.gen_save_path, 'generated_images', self.cond_type, self.pure_img_path)

        # original img path
        self.ori_img_path = os.path.join(self.dataset_path, self.rel_ood_img_path)

        # unc save path
        self.subname = extract_substring(rel_ood_img_path)
        self.unc_save_path = os.path.join(self.save_path, 'unc_results', self.cond_type, f'{self.subname}_{self.upsampler_name}.pt')
        self.base_save_path = os.path.join(self.save_path, 'unc_results', self.cond_type, f'{self.subname}_basics.pt')

        # visualization fig save path
        self.fig_save_path = os.path.join(self.save_path, 'visualization', self.cond_type, f'{self.subname}_{self.upsampler_name}.png')

        input_size = 224
        self.transform = T.Compose([
            T.Resize(input_size),
            T.CenterCrop((input_size, input_size)),
            T.ToTensor(),
        ])
    

    def get_generated_img(self, diffusion, save_image = True, overwrite = False):
        # check if the image is already generated
        if not overwrite and glob.glob(self.gen_img_path):
            if self.verbose: print('already generated at', self.gen_img_path)
            return
        
        # generate image
        if self.verbose: print('diffusion img inside get generated img', self.dataset_path)
        ood_img = load_input_img(self.ori_img_path, image_size = self.image_size)
        x_ood = diffusion.encode(ood_img)
        if self.ddim:
            # print('ddim sampling')
            img = diffusion.ddim_sample_with_conditioning(
                y_start = x_ood,
                t_begin = 999,
                w = self.ddim_guidance_scale,
                sampling_timesteps = self.ddim_sampling_timesteps,
                return_all_timesteps= False)
        else:
            for x in diffusion.p_sample_loop(
                batch_size=1,
                cond_fn=self.conditioner,
                guidance_kwargs=dict(x_targ=x_ood)
            ):
                img = diffusion.decode(x)  # (1, 3, H, W)
            
        final_img: Tensor = img  # (1, 3, H, W)
        if save_image:
            image = T.ToPILImage()(final_img[0])
            image.save(self.gen_img_path)
        return 

    def run_pipe(self, diffusion, mask_generator, overwrite = False):
        # check if .pt already exist first
        # print('gen img path',self.base_save_path )
        # print('overwrite', overwrite)
        start = time.time()
        ######## image generation ######## 
        if not overwrite and glob.glob(self.base_save_path):
            # check if the ood, ind and seg tensors are already generated
            if self.verbose: print('baisc ts already saved at', self.base_save_path)
            base_dic = torch.load(self.base_save_path)
            ood_img, ind_img, seg_ts = base_dic['ood'], base_dic['ind'], base_dic['seg']
            if self.square: # use orignal image size for segmentation
                ood_img_pil = Image.open(self.ori_img_path).convert("RGB")
            else:
                ood_img_pil = T.ToPILImage()(ood_img[0])
            seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3, verbose = False)
        else:
             # generate img from diffusion
            self.get_generated_img(diffusion, overwrite = overwrite)
        
            if self.square:
                ood_img = load_input_img(self.ori_img_path, no_cuda=True)[0]
                ind_img = load_input_img(self.gen_img_path, no_cuda=True)[0]

                ood_img_pil = Image.open(self.ori_img_path).convert("RGB")
                seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3, verbose = False)
                seg_im = show_image_and_masks(ood_img_pil, seg, bg_alpha=0.1, fg_alpha=0.90)
                seg_ts = T.ToTensor()(seg_im)

                diff_l2 = lambda x, y: (x - y).square().sum(dim=0)  # (1, 3, H, W) -> (1, H, W)
                diff_cos = lambda x, y: 1-F.cosine_similarity(x,y, dim = 0) # (1, 3, H, W) -> (1, H, W)
                upsample = torch.nn.Upsample(size= self.ori_image_size, mode='bilinear', align_corners=False)
                pixel_up = lambda ind_img: F.interpolate(ind_img.unsqueeze(0), size=self.ori_image_size, mode='bilinear', align_corners=False)[0]
                diff_pixel = upsample(diff_l2(ood_img, pixel_up(ind_img)).unsqueeze(0).unsqueeze(0))[0][0]
                diff_pixel_cos  = upsample(diff_cos(ood_img, pixel_up(ind_img)).unsqueeze(0).unsqueeze(0))[0][0]
                base_dic = {'ood': ood_img, 'ind': ind_img, 'seg': seg_ts, 'diff_norm': diff_pixel, 'diff_cos': diff_pixel_cos}
                torch.save(base_dic, self.base_save_path)

            else:
                ood_img = self.transform(Image.open(self.ori_img_path).convert("RGB")).unsqueeze(0).cuda()
                ind_img = self.transform(Image.open(self.gen_img_path).convert("RGB")).unsqueeze(0).cuda()
                
                ood_img_pil = T.ToPILImage()(ood_img[0])
                seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3, verbose = self.verbose)
                seg_im = show_image_and_masks(ood_img_pil, seg, bg_alpha=0.1, fg_alpha=0.90)
                seg_ts = T.ToTensor()(seg_im)

                # diff in pixel space
                original, generated = ood_img[0].detach().cpu(), ind_img[0].detach().cpu()
                diff=(original/255-generated/255).abs().float()  # (1, 3, H, W)
                diff_norm: Tensor = diff.norm(dim=0)
                diff_prod: Tensor = diff.prod(dim=0)  # (1, H, W)
                diff_softrect: Tensor = 1 - softrect_inlier(original/255, generated/255, width=0.1, steepness=100).prod(dim=0)  # (1, H, W)
                # cos pixel diff
                diff_cos = lambda x, y: 1-F.cosine_similarity(x,y) # (1, 3, H, W) -> (1, H, W)
                diff_pixel_cos:  Tensor = diff_cos(ood_img, ind_img)  # (1, H, W)
                # save basic dics
                base_dic = {'ood': ood_img, 'ind': ind_img, 'seg': seg_ts, 'diff_norm': diff_norm, 'diff_prod': diff_prod, 'diff_softrect': diff_softrect, 'diff_cos': diff_pixel_cos}
                torch.save(base_dic, self.base_save_path)

        if TIME:
            end_process_1 = time.time()  # End of process 1
            print(f"Time for image generation: {end_process_1 - start:.2f} seconds")       
        ######## upfeat ######## 
        if not self.only_generate_img:
            if not overwrite and glob.glob(self.unc_save_path):
                if self.verbose: print('unc results already generated at', self.unc_save_path)
                if self.ssim:
                    ssim_diff, ssim_diff_sam = calculate_ssim(ood_img, ind_img, seg, self.ori_image_size, self.image_size)
                    ts_dic = torch.load(self.unc_save_path)
                    ts_dic['ssim'] = ssim_diff
                    ts_dic['ssim_sam'] = ssim_diff_sam
                    torch.save(ts_dic, self.unc_save_path)
                return

            # get diff with the specified upfeat models
            upsample = torch.nn.Upsample(size=self.ori_image_size, mode='bilinear', align_corners=False)
            pixel_up = lambda ind_img: F.interpolate(ind_img.unsqueeze(0), size=self.ori_image_size, mode='bilinear', align_corners=False)[0]
            # ood_img, ind_img, upsampler, seg, upsample
            diff_list = experiment(ood_img, ind_img, self.upsampler, seg, upsample, pixel_up, self.square)
            # save img
            ts_dic = dict()
            for ts, ts_name in zip(diff_list,['LowRes', 'HighRes', 'LowRes_cos', 'HighRes_cos','HighRes_cos_sqr',
                    'SAM_Pixel', 'SAM_LowRes', 'SAM_HighRes', 'SAM_Pixel_cos', 'SAM_LowRes_cos', 'SAM_HighRes_cos', 'SAM_HighRes_cos_sqr']):
                ts_dic[ts_name] = ts
            torch.save(ts_dic, self.unc_save_path)
        if TIME:
            end_process_2 = time.time()  # End of process 2
            featup = end_process_2 - end_process_1
            print(f"Time for featup with SAM: {featup:.2f} seconds")
            total_time = time.time() - start
            print(f"Total execution time: {total_time:.2f} seconds")
        return 

    def visualize(self, save):
        base_dic = torch.load(self.base_save_path)
        unc_dic = torch.load(self.unc_save_path)
        num_rows = 3
        fig = plt.figure(figsize=(16, 3.5*num_rows))
        pixel_up = lambda ind_img: F.interpolate(ind_img.unsqueeze(0), size=self.ori_image_size, mode='bilinear', align_corners=False)[0]
        plot_all_diff(base_dic['ood'], pixel_up(base_dic['ind']), base_dic['seg'], num_rows)
        diff_subplot(2, num_rows, self.upsampler_name, unc_dic, square = self.square)
        diff_subplot(3, num_rows, self.upsampler_name, unc_dic, square = self.square,
                     title_dic = {'LowRes_cos': 'cos Low-Res Feats', 
                 'HighRes_cos': 'cos High-Res Feats',
                 'SAM_Pixel_cos': 'SAM cos Pixel-space', 
                 'SAM_LowRes_cos': 'SAM cos Low-Res Feats', 
                 'SAM_HighRes_cos': 'SAM cos High-Res Feats'})

        title = f"{self.cond_type} {self.upsampler_name} {self.subname}"
        plt.suptitle(title, y = 0.92)
        plt.show()
        if save:
            fig.savefig(self.fig_save_path)




        

class Pipeline():
    def __init__(self, diffusion_model_name, dataset_path, save_path, image_size, conditioners, upsample_models, mask_generator, ori_image_size, progress_bar = True,ssim = False,
                 verbose = False, report_num = 1, visualize_num = 20, overwrite = False, only_generate_img = False, square = True, gen_save_path = None, model_path = None,
                 ddim = False, ddim_sampling_timesteps = 250, ddim_guidance_scale = 0.5,
                 quantize = False, quant_save_path = None, quant_bits = 16):
        self.diffusion_model_name = diffusion_model_name
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.image_size  = image_size
        self.ori_image_size = ori_image_size
        self.conditioners = conditioners
        self.upsample_models = upsample_models
        self.mask_generator = mask_generator
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.report_num = report_num
        self.visualize_num = visualize_num
        self.overwrite = overwrite # overwrite unc maps if already exist
        self.only_generate_img = only_generate_img #only generate diffusion images, but not get unc map
        self.square = square # do the cutting into square and merging for better alignment
        if not gen_save_path: gen_save_path = save_path
        self.gen_save_path = gen_save_path
        self.model_path = model_path
        self.ssim = ssim #run_ssim
        self.ddim = ddim
        self.ddim_sampling_timesteps = ddim_sampling_timesteps
        self.ddim_guidance_scale = ddim_guidance_scale
        self.quant_save_path = quant_save_path
        self.quantize = quantize
        self.quant_bits = quant_bits

        ###### load diffusion model ######
        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True
        )

        diffusion = GaussianDiffusion(
            model,
            image_size = self.image_size,    # size of image (height, width) originally this was (128, 128)
            timesteps = 1000,           # number of steps
            sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            objective = 'pred_v',       # note that default is different for guided_diffusion vs denoising_diffusion_pytorch
            progress_bar = self.progress_bar
        ).cuda()
        if not self.model_path:
            # load model from checkpoint
            model_path = download_from_cloud(self.diffusion_model_name)
        print(model_path)
        state_dict = torch.load(model_path)
        diffusion.load_state_dict(state_dict) #['model']

        if self.quantize:
            from quantizer import quantize_model
            quantized_state_dict = quantize_model(diffusion, self.quant_bits)

            quantized_model_name = self.diffusion_model_name.replace('model-', f'model-{self.quant_bits}bits-')
            torch.save(quantized_state_dict, f"{self.quant_save_path}/{quantized_model_name}")
            print('quantized model saved at', self.quant_save_path)
            diffusion.load_state_dict(quantized_state_dict)

        self.diffusion = diffusion
        if self.verbose:
            # display samples from model produced during training
            sample_path = download_from_cloud(self.diffusion_model_name.replace('model-', 'sample-').replace('.pt', '.png'))
            IPyImage(sample_path, width=700),display()

        # get conditioners AFTER loading the diffusion model
        self.cond_dic = {
        'l2': Conditioner(
                diffusion = self.diffusion,
                objectives=[
                    L2Objective(guidance_scale=7e4),
                ],
            ),
        'l2_q': Conditioner(
                diffusion = self.diffusion,
                objectives=[
                    L2Objective(guidance_scale=1e5),
                ],
                apply_q_transform=True,
            ),
        'l2_p':  Conditioner(
            diffusion=diffusion,
            objectives=[
                L2Objective(guidance_scale=1e4),
            ],
            apply_p_transform=True,
        ),
        'exp_linear': Conditioner(
                diffusion = self.diffusion,
                objectives=[
                    ExponentialPriorSPMObjective(guidance_scale=3e3),
                ],
                params=ScheduledParams(
                        width=(2.0,   0.1, 'linear'),
                    steepness=(0.1, 100.0, 'linear'),
                            λ=( 20,    20, 'fixed' ),
                ),
            ),
        'exp_linear_q': Conditioner(
                diffusion = self.diffusion,
                objectives=[
                    ExponentialPriorSPMObjective(guidance_scale=3e3),
                ],
                params=ScheduledParams(
                        width=(2.0,   0.1, 'linear'),
                    steepness=(0.1, 100.0, 'linear'),
                            λ=( 20,    20, 'fixed' ),
                ),
                apply_q_transform=True,
            ),
        'exp_linear_p': Conditioner(
                diffusion = self.diffusion,
                objectives=[
                    ExponentialPriorSPMObjective(guidance_scale=5e2),
                ],
                params=ScheduledParams(
                        width=(2.0,   0.1, 'linear'),
                    steepness=(0.1, 100.0, 'linear'),
                            λ=( 20,    20, 'fixed' ),
                ),
                apply_p_transform=True,
            )

        }

        guidance_scales = [7e3, 1e4, 2e4, 3e4, 5e4, 1e5, 7e4, 3e5]
        guidance_scales_str = ['7e3', '1e4', '2e4', '3e4', '5e4', '1e5', '7e4', '3e5']
        l2_p_dict = {f'l2_p_{guidance_str}': Conditioner(
                    diffusion=self.diffusion,
                    objectives=[
                        L2Objective(guidance_scale=guidance_scale),
                    ],
                    apply_p_transform=True,
                ) for guidance_scale, guidance_str in zip(guidance_scales, guidance_scales_str)}
        guidance_scales = [5e5]
        guidance_scales_str = ['5e5']
        l2_q_dict = {f'l2_q_{guidance_str}': Conditioner(
                    diffusion=self.diffusion,
                    objectives=[
                        L2Objective(guidance_scale=guidance_scale),
                    ],
                    apply_q_transform=True,
                ) for guidance_scale, guidance_str in zip(guidance_scales, guidance_scales_str)}

        guidance_scales = [5e4, 7e4, 3e2, 4e2, 6e2, 7e2, 1e3]
        guidance_scales_str = ['3e2', '4e2', '6e2', '7e2', '1e3']
        exp_p_dict = {f'exp_linear_p_{guidance_str}': Conditioner(
                        diffusion = self.diffusion,
                        objectives=[
                            ExponentialPriorSPMObjective(guidance_scale=guidance_scale),
                        ],
                        params=ScheduledParams(
                                width=(2.0,   0.1, 'linear'),
                            steepness=(0.1, 100.0, 'linear'),
                                    λ=( 20,    20, 'fixed' ),
                        ),
                        apply_p_transform=True,
                    ) for guidance_scale, guidance_str in zip(guidance_scales, guidance_scales_str)}
        
        guidance_scales = [3e3]
        guidance_scales_str = ['3e3']
        exp_q_dict = {f'exp_linear_q_{guidance_str}': Conditioner(
                        diffusion = self.diffusion,
                        objectives=[
                            ExponentialPriorSPMObjective(guidance_scale=guidance_scale),
                        ],
                        params=ScheduledParams(
                                width=(2.0,   0.1, 'linear'),
                            steepness=(0.1, 100.0, 'linear'),
                                    λ=( 20,    20, 'fixed' ),
                        ),
                        apply_q_transform=True,
                    ) for guidance_scale, guidance_str in zip(guidance_scales, guidance_scales_str)}
        self.cond_dic = self.cond_dic | l2_p_dict | exp_p_dict | l2_q_dict |exp_q_dict
        

    def run(self, image_paths = None):
        # Call the function
        if not image_paths:
            image_paths = sorted(get_image_paths(self.dataset_path))
        for cond_type in self.conditioners:
            if self.ddim: # if ddim, only use l2 conditioner
                conditioner = None
            else:
                conditioner = self.cond_dic[cond_type]
            for upsample_model in self.upsample_models:
                count = 0
                use_norm = (upsample_model!='maskclip') # if 'maskclip', False
                upsampler = torch.hub.load("mhamilton723/FeatUp", upsample_model, use_norm = use_norm).to('cuda')
                for rel_ood_img_path in image_paths:
                    p = OneImgPipe(dataset_path = self.dataset_path, save_path = self.save_path, rel_ood_img_path = rel_ood_img_path, 
                                image_size = self.image_size, cond_type = cond_type, conditioner = conditioner, 
                                upsampler= upsampler, upsampler_name = upsample_model, verbose = self.verbose, 
                                only_generate_img = self.only_generate_img,  square = self.square, ori_image_size = self.ori_image_size,
                                gen_save_path = self.gen_save_path, ssim = self.ssim, 
                                ddim =  self.ddim, ddim_sampling_timesteps = self.ddim_sampling_timesteps, ddim_guidance_scale = self.ddim_guidance_scale)
                    p.run_pipe(self.diffusion, self.mask_generator, overwrite = self.overwrite )
                    if count%self.report_num == 0:
                        print('progress:', count, p.rel_ood_img_path, upsample_model, cond_type)
                    if count% (self.visualize_num*5) == 0:
                        p.visualize(save = True)
                        print('subresult visualized', p.rel_ood_img_path, upsample_model, cond_type)
                    count += 1
   
            # for vid in np.arange(0, len(image_paths), self.visualize_num):
                # self.visualize(image_paths[vid], cond_type)
            

    def visualize(self, rel_ood_img_path, cond_type, save = True):
        num_rows = len(self.upsample_models)*2 + 1
        fig = plt.figure(figsize=(16, 3*num_rows))
        subname = extract_substring(rel_ood_img_path)
        base_save_path = os.path.join(self.save_path, 'unc_results', cond_type, f'{subname}_basics.pt')
        base_dic = torch.load(base_save_path)
        pixel_up = lambda ind_img: F.interpolate(ind_img.unsqueeze(0), size=self.ori_image_size, mode='bilinear', align_corners=False)[0]
        plot_all_diff(base_dic['ood'], pixel_up(base_dic['ind']), base_dic['seg'], num_rows)
        for i, upsample_model in enumerate(self.upsample_models):
            unc_save_path = os.path.join(self.save_path, 'unc_results', cond_type, f'{subname}_{upsample_model}.pt')
            dic = torch.load(unc_save_path)
            diff_subplot(i*2 + 2, num_rows, upsample_model, dic)
            diff_subplot(i*2 + 3, num_rows, upsample_model, dic, title_dic = {'LowRes_cos': 'cos Low-Res Feats', 
                 'HighRes_cos': 'cos High-Res Feats',
                 'SAM_Pixel_cos': 'SAM cos Pixel-space', 
                 'SAM_LowRes_cos': 'SAM cos Low-Res Feats', 
                 'SAM_HighRes_cos': 'SAM cos High-Res Feats'})
        plt.suptitle(f"{cond_type} {subname}", y = 0.92)
        plt.show()
        if save:
            fig_save_path = os.path.join(self.save_path, 'visualization', cond_type, f'{subname}.png')
            fig.savefig(fig_save_path, dpi=300)
                               

if __name__ == '__main__':
    pass