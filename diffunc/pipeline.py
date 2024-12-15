import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from IPython.display import Image as IPyImage
from IPython.display import display
import jupyviz as jviz
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from featup.util import norm, unnorm, pca
from featup.plotting import plot_feats, plot_lang_heatmaps

import os
import glob

from denoising_diffusion_pytorch import Unet
from diffunc import viz
from diffunc.sam import Segmentation, show_image_and_masks
from diffunc.all import (
    GaussianDiffusion,
    ScheduledParams,
    Conditioner,
    L2Objective,
    ExponentialPriorSPMObjective,
    set_random_seed,
    load_input_img,
    download_from_cloud,
    softrect_inlier,
)

def load_diffusion_model(dataset, model_name, display_image = False):
    if dataset == 'RUGD':
        image_size = (176, 216)
    elif dataset == 'cityscapes':
        image_size = (256, 512)
    else:
        raise NameError('Invalid datset', dataset)
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,    # size of image (height, width) originally this was (128, 128)
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = 'pred_v',       # note that default is different for guided_diffusion vs denoising_diffusion_pytorch
        use_vae = False,            # whether to use latent space
    ).cuda()

    # load model from checkpoint
    model_path = download_from_cloud(model_name)
    state_dict = torch.load(model_path)
    diffusion.load_state_dict(state_dict['model'])
    sample_path = download_from_cloud(model_name.replace('model-', 'sample-').replace('.pt', '.png'))
    print('diffusion model sample images path:', sample_path)
    if display_image:
        img = Image.open(sample_path)
        display(img)
    return diffusion

def diff_subplot(row, num_rows, upsample_model, diff_list):
    title_list = ['L2 Low-Res Feats', 'L2 High-Res Feats',
                  'SAM L2 Pixel-space', 
                  'SAM L2 Low-Res Feats', 'SAM L2 High-Res Feats']
    for idx, diff in enumerate(diff_list):
        # if (row-1)*5+idx+1>25:
        #     print('row, idx', row, idx)
        ax1 = plt.subplot(num_rows, 5, (row-1)*5+idx+1)
        plt.imshow(diff[0].detach().cpu())
        ax1.set_title(title_list[idx])
        if idx==0:
            ax1.set_ylabel(upsample_model)
    
def plot_all_diff(ood_img, ind_img, seg_ts, num_rows):
    ### need to DIVIDE BY 255 here to restore the same effect as in in prev notebooks since not normalized here
    original, generated = ood_img[0].detach().cpu(), ind_img[0].detach().cpu()
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
    
@torch.no_grad()
def experiment(ood_img, ind_img, upsampler, seg, size=None):
    display_width = 300

    maybe_resize = T.Resize(size) if size is not None else nn.Identity()
    ood_input = maybe_resize(ood_img)  # (1, 3, H, W)
    ind_input = maybe_resize(ind_img)  # (1, 3, H, W)

    # ood feats
    ood_hr_feats = upsampler(norm(ood_input))
    ood_lr_feats = upsampler.model(norm(ood_input))

    # ind feats
    ind_hr_feats = upsampler(norm(ind_input))
    ind_lr_feats = upsampler.model(norm(ind_input))

    # perform PCA on all features so that they can all be compared
    [ood_hr_feats_pca, ood_lr_feats_pca, ind_hr_feats_pca, ind_lr_feats_pca], _ = \
        pca([ood_hr_feats, ood_lr_feats, ind_hr_feats, ind_lr_feats])

    # # viz OOD feats
    # imgs = [ood_input, ood_lr_feats_pca, ood_hr_feats_pca]
    # titles = ['OOD Input Image', 'Original Featuress', 'Upsampled Features']
    # jviz.tile([jviz.img(e) for e in imgs]).html(titles=titles, width=display_width).display();

    # # viz IND feats
    # imgs = [ind_input, ind_lr_feats_pca, ind_hr_feats_pca]
    # titles = ['ID Input Image', 'Original Featuress', 'Upsampled Features']
    # jviz.tile([jviz.img(e) for e in imgs]).html(titles=titles, width=display_width).display();

    # compute L2 distances
    diff_l2 = lambda x, y: (x - y).square().sum(dim=1)  # (1, 3, H, W) -> (1, H, W)
    diff_pixel = diff_l2(ood_input, ind_input)  # (1, H, W)
    diff_lr = diff_l2(ood_lr_feats, ind_lr_feats)  # (1, H, W)
    diff_hr = diff_l2(ood_hr_feats, ind_hr_feats)  # (1, H, W)
    # imgs = [viz.colormap(e) for e in [diff_pixel, diff_lr, diff_hr]]
    # titles = ['L2 Pixel-space', 'L2 Low-Res Feats', 'L2 High-Res Feats']
    # jviz.tile([jviz.img(e) for e in imgs]).html(titles=titles, width=display_width).display();


    # # postprocess using SAM
    # imgs = [viz.colormap(seg.regularize(e, fill=0)) for e in [diff_pixel, diff_lr, diff_hr]]
    # titles = ['SAM-Regularized L2 Pixel-space', 'SAM-Regularized L2 Low-Res Feats', 'SAM-Regularized L2 High-Res Feats']
    # jviz.tile([jviz.img(e) for e in imgs]).html(titles=titles, width=display_width).display();
    return [diff_lr, diff_hr] + [seg.regularize(e, fill=0) for e in [diff_pixel, diff_lr, diff_hr]]

def extract_substring(s):
    if '/' in s:
        # Find the position of the last '/' and '.png'
        pos_slash = s.rfind('/') + 1  # we add 1 to start after the '/'
    else:
        pos_slash =  0
    if '.png' in s:
        pos_png = s.rfind('.png')
    if '.jpg' in s:
        pos_png = s.rfind('.jpg')
    
    # Extract the substring between these positions
    result = s[pos_slash:pos_png]
    return result

def get_generated_img(rel_ood_img_path, diffusion, img_parent_path, conditioner: Conditioner, iview_active: bool=True, 
                      save_image = True, folder = 'l2_1e5', parent_path = ''):
    image_size = (176, 216) #(256, 512)
    print('inside get generated img', img_parent_path)
    ood_img = load_input_img( img_parent_path + rel_ood_img_path, image_size = image_size)
    x_ood = diffusion.encode(ood_img)
    print(x_ood.shape)

    with jviz.iview(active = False, window_title='Reverse diffusion') as iview:
        # run guided reverse diffusion
        revdiff = []
        for x in diffusion.p_sample_loop(
            batch_size=1,
            cond_fn=conditioner,
            guidance_kwargs=dict(x_targ=x_ood),
        ):
            img = diffusion.decode(x)  # (1, 3, H, W)
            iview.update(img)
            revdiff.append(img)  # List[(Tensor, shape=(1, 3, H, W))]

    final_img: Tensor = revdiff[-1]  # (1, 3, H, W)
    if save_image:
        image = T.ToPILImage()(final_img[0])
        image.save(f'{parent_path}generated_images/{folder}/{extract_substring(rel_ood_img_path)}.png')
    return final_img

def unc_pipe(rel_ood_img_path, diffusion, conditioner, mask_generator, folder, 
             input = None,
             cityscapes = False, parent_path = '',
             img_parent_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/ood/',
             overwrite = False):
    folder_path = f'generated_images/{folder}'
    search_pattern = f'{extract_substring(rel_ood_img_path)}.png'  # This pattern can be changed to match different file types or naming patterns
    subname = extract_substring(rel_ood_img_path)
    if overwrite or not glob.glob(os.path.join(parent_path, folder_path, search_pattern)):
        # generate image
        if cityscapes:
            size = (256, 512)
        else:
            size = (176, 216)
        print(img_parent_path)
        final_img = get_generated_img(rel_ood_img_path, diffusion, img_parent_path, conditioner, folder=folder, parent_path = parent_path)
    else:
        print('image already generated')
    
    # input image transformation
    input_size = 224
    transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop((input_size, input_size)),
        T.ToTensor(),
    ])
    
    ood_img_path = img_parent_path + rel_ood_img_path
    # if input:

    ood_img = transform(Image.open(ood_img_path).convert("RGB")).unsqueeze(0).cuda()

    ind_img_path = f'{parent_path}generated_images/{folder}/{extract_substring(rel_ood_img_path)}.png'
    ind_img = transform(Image.open(ind_img_path).convert("RGB")).unsqueeze(0).cuda()
    
    
    #segmentation
    ood_img_pil = T.ToPILImage()(ood_img[0])
    seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3)
    seg_im = show_image_and_masks(ood_img_pil, seg, bg_alpha=0.1, fg_alpha=0.90)
    seg_ts = T.ToTensor()(seg_im)
    all_dic = dict()
    all_dic['basics'] = {'ood': ood_img, 'ind': ind_img, 'seg': seg_ts}
    # get diff with diff upfeat models, visualize
    num_rows = 7
    fig = plt.figure(figsize=(16, 3*num_rows))
    plot_all_diff(ood_img, ind_img, seg_ts, 6)
    
    for i, upsample_model in enumerate(['dino16', 'dinov2','clip', 'vit', 'resnet50', 'maskclip']):
        use_norm = True
        if upsample_model == 'maskclip':
            use_norm = False
        upsampler = torch.hub.load("mhamilton723/FeatUp", upsample_model, use_norm=use_norm).to('cuda')
        diff_list = experiment(ood_img, ind_img, upsampler, seg)
        diff_subplot(i + 2,  num_rows, upsample_model, diff_list)
        # save img
        ts_dic = dict()
        for ts, ts_name in zip(diff_list,['LowRes', 'HighRes',
                  'SAM_Pixel', 'SAM_LowRes', 'SAM_HighRes']):
            ts_dic[ts_name] = ts
        all_dic[upsample_model] = ts_dic
    torch.save(all_dic, f'/home/sunsh16e/diffunc/experiment/rugd/unc_results/{folder}/{subname}.pt')
            
    plt.suptitle(f"{folder}_{subname}", y = 0.92)
    plt.show()
    fig.savefig(f'{parent_path}results/{folder}/{subname}.png', dpi=300)

def compare_pipe(rel_ood_img_path, conditioners, models, diffusion, mask_generator, save_folder):
    cond_dic = {
    'l2_q': Conditioner(
            diffusion=diffusion,
            objectives=[
                L2Objective(guidance_scale=1e5),
            ],
            apply_q_transform=True,
        ),
    'exp_linear_q': Conditioner(
            diffusion=diffusion,
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
            diffusion=diffusion,
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
    
    num_cols = len(conditioners)+1
    num_rows = len(models) +1

    # input image transformation
    input_size = 224
    transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop((input_size, input_size)),
        T.ToTensor(),
    ])

    # ood image
    ood_img_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/ood/' + rel_ood_img_path
    ood_img = transform(Image.open(ood_img_path).convert("RGB")).unsqueeze(0).cuda()
    ## segmentation
    ood_img_pil = T.ToPILImage()(ood_img[0])
    seg = Segmentation(mask_generator, ood_img_pil, resize=None, min_mask_area_frac=1e-3)
    seg_im = show_image_and_masks(ood_img_pil, seg, bg_alpha=0.1, fg_alpha=0.90)
    seg_ts = T.ToTensor()(seg_im)

    ## visualize
    original = ood_img[0].detach().cpu()


    fig = plt.figure(figsize=(16, 3.1*num_rows))
    ax1 = plt.subplot(num_rows, num_cols, 1)
    plt.imshow(original.permute(1, 2, 0))
    ax1.set_title(extract_substring(rel_ood_img_path))

    # diffusion generated image
    for cond_id, cond_type in enumerate(conditioners):
        # Set up the conditioner
        conditioner = cond_dic[cond_type]
        folder_path = f'generated_images/{cond_type}'
        search_pattern = f'{extract_substring(rel_ood_img_path)}.png'  # This pattern can be changed to match different file types or naming patterns

        if not glob.glob(os.path.join(folder_path, search_pattern)):
            # generate image
            final_img = get_generated_img(rel_ood_img_path, diffusion, conditioner, folder=cond_type)
        else:
            print('image already generated')

        ## visualize
        ind_img_path = f'generated_images/{cond_type}/{extract_substring(rel_ood_img_path)}.png'
        ind_img = transform(Image.open(ind_img_path).convert("RGB")).unsqueeze(0).cuda()
        
        generated = ind_img[0].detach().cpu()
        
        ax2 = plt.subplot(num_rows, num_cols, cond_id+2)
        plt.imshow(generated.permute(1, 2, 0))
        ax2.set_title(cond_type)
        
        for model_idx, upsample_model in enumerate(models):
            upsampler = torch.hub.load("mhamilton723/FeatUp", upsample_model, use_norm=True).to('cuda')
            diff_lr, diff_hr, sam_diff_pixel, sam_diff_lr, sam_diff_hr = experiment(ood_img, ind_img, upsampler, seg)
            
            ax1 = plt.subplot(num_rows, num_cols, cond_id + (model_idx+1)*4 +2)
            plt.imshow(sam_diff_hr[0].detach().cpu())
            # ax1.set_title(upsample_model)
            if cond_id==0:
                ax1.set_ylabel(upsample_model) #'SAM L2 High-Res Feats')
    
    fig.savefig(f'{save_folder}/{extract_substring(rel_ood_img_path)}.png', dpi=150)


