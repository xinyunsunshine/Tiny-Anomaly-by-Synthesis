import torch
from torch import Tensor
from IPython.display import Image as IPyImage
import torchvision.transforms as T
import jupyviz as jviz
import os

from denoising_diffusion_pytorch import Unet

from diffunc import viz
from diffunc.all import (
    get_image_paths,
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
)

# always call this first
set_random_seed(0)

model_name = 'rugd_full_id_train/model-10.pt'

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = (152, 240), #(176, 216),    # size of image (height, width) originally this was (128, 128)
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective = 'pred_v',       # note that default is different for guided_diffusion vs denoising_diffusion_pytorch
).cuda()

# load model from checkpoint
conditioner_name = 'l2p'
model_num = 50
model_path = '/home/sunsh16e/diffunc/models/rellis_full_id/model-50.pt' #download_from_cloud(model_name)
state_dict = torch.load(model_path)
diffusion.load_state_dict(state_dict['model'])

# dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
dataset_path = '/home/sunsh16e/diffunc/data/RELLLIS/examples'
image_path_name = get_image_paths('/home/sunsh16e/diffunc/data/RELLLIS/examples')
# image_path_name = [
#     # "trail-7/trail-7_00011.png",
#     # "trail-7/trail-7_00331.png",
#     # "trail-13/trail-13_00146.png",
#     # "trail-13/trail-13_00101.png",
#     # "park-8/park-8_00501.png",
#     # "trail-7/trail-7_00426.png",
#     # "trail-13/trail-13_00556.png",
#     # "trail-13/trail-13_00766.png",
#     # "park-8/park-8_00076.png",


#     "trail/trail_00896.png",
#     "trail-15/trail-15_00096.png",
#     "creek/creek_00001.png",
#     "creek/creek_00166.png",
#     "park-1/park-1_01911.png",
#     "park-1/park-1_00736.png",
#     "park-2/park-2_01616.png",
#     "park-2/park-2_02586.png",
#     "park-8/park-8_01706.png",
#     "trail-3/trail-3_02621.png",
#     "trail-4/trail-4_00641.png",
#     "trail-6/trail-6_01171.png",
#     "trail-6/trail-6_01501.png",
#     "trail-7/trail-7_00251.png",
#     "trail-7/trail-7_00531.png",
#     "trail-7/trail-7_00996.png",
#     "trail-11/trail-11_01571.png",
#     "trail-11/trail-11_00871.png",
#     "trail-13/trail-13_00566.png",
#     "trail-14/trail-14_00611.png",
#     ]
for rel_ood_img_path in image_path_name:
    print(rel_ood_img_path)
    ood_img_path = os.path.join(dataset_path, rel_ood_img_path)
    ood_img = load_input_img(ood_img_path, image_size=(152, 240))#176, 216))
    x_ood = diffusion.encode(ood_img)

    def run_and_viz(conditioner: Conditioner, iview_active: bool=True):
        with jviz.iview(iview_active, window_title='Reverse diffusion') as iview:
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

        # display reverse diffusion GIFs
        # indices = torch.linspace(0, 1000, 25, dtype=torch.int)
        # img = jviz.img(ood_img)
        # gif = jviz.gif([revdiff[ind] for ind in indices], time_in_ms=2000, hold_last_frame_time_in_ms=1000)
        # display(jviz.tile([img, gif]).html(width=300))

        # visualize diff image i.e. uncertainty
        final_img: Tensor = revdiff[-1]  # (1, 3, H, W)
        # diff: Tensor = (final_img - ood_img).abs()  # (1, 3, H, W)
        # diff_norm: Tensor = diff.norm(dim=1)  # (1, H, W)
        # diff_prod: Tensor = diff.prod(dim=1)  # (1, H, W)
        # diff_softrect: Tensor = 1 - softrect_inlier(final_img, ood_img, width=0.1, steepness=100).prod(dim=1)  # (1, H, W)
        # diff_imgs = [viz.colormap(e) for e in [diff_norm, diff_prod, diff_softrect]]
        # titles = ['Diff-Norm', 'Diff-Prod', 'Diff-Softrect']
        # jviz.show_images([*diff_imgs, ood_img, final_img], titles=[*titles, 'Input', 'Generated'], columns=3, width=250)
        return final_img

    for guidance_scale in [5e4, 7e4, 1e5, 3e5, 5e5]: #[1e2, 3e2, 5e2, 7e2, 1e3]: #[1e4, 3e4, 5e4, 1e5]: #[1e5, 7e4, 5e4, 3e4, 1e4]:
        # Set up the conditioner
        conditioner = Conditioner(
            diffusion=diffusion,
            objectives=[
                L2Objective(guidance_scale=guidance_scale),
            ],
            apply_p_transform=True,
        )

        # conditioner =  Conditioner(
        #         diffusion = diffusion,
        #         objectives=[
        #             ExponentialPriorSPMObjective(guidance_scale=guidance_scale), #5e2
        #         ],
        #         params=ScheduledParams(
        #                 width=(2.0,   0.1, 'linear'),
        #             steepness=(0.1, 100.0, 'linear'),
        #                     λ=( 20,    20, 'fixed' ),
        #         ),
        #         apply_p_transform=True,
        #     )

        final_img = run_and_viz(conditioner)
        image = T.ToPILImage()(final_img[0])
        image.save(f'/home/sunsh16e/diffunc/experiment/rellis/{conditioner_name}_exp_{model_num}/generated_images/{rel_ood_img_path[:-4]}_{conditioner_name}_{guidance_scale}.png')
        # image.save(f'experiment/rugd/l2_exp/{extract_substring(rel_ood_img_path)}_l2_{guidance_scale}.png')