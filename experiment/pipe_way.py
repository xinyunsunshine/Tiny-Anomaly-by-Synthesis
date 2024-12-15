from pipe import Pipeline
from diffunc.all import load_sam, get_image_paths, set_random_seed
import os
# always call this first
# set_random_seed(0)

square = True
only_generate_img = False
overwrite = True # normally set as False
verbose = False
report_num = 10
visualize_num = 20
dataset = 'rugd'
ssim = False
sp = False

######## DDIM #####
ddim = True
ddim_sampling_timesteps_list = [500] # 250, 500 8510, 8528, 8654, 8844
ddim_guidance_scale = 0.6
###################
if sp:
    dir = '/home/gridsan/sjiang/diffunc/experiment'
    data_dir = '/home/gridsan/groups/rrg/data' #/RELLIS/ddpm_train_sets'
    rellis_model_dir = '/home/gridsan/groups/rrg/results/rellis_full_id'
    progress_bar = False
else:
    dir = '/home/sunsh16e/diffunc/experiment'
    data_dir = '/home/sunsh16e/diffunc/data'
    rellis_model_dir = '/home/sunsh16e/diffunc/models/rellis_full_id'
    progress_bar = True

# load diffusion model
if dataset == 'rugd':
    model_path = "/home/sunsh16e/diffunc/cloud/rugd_full_id_train/naive_quant_model.pt"
    diffusion_model_name = 'rugd_full_id_train/model-10.pt'
    dataset_path = f'{data_dir}/RUGD/ddpm_train_sets/full/images/ood'
    gen_save_path = f'{dir}/rugd'
    image_size = (176, 216)
    ori_image_size = (550, 688)
    if square:
        save_path = f'{dir}/rugd_sqr_new'
    else:
        save_path = f'{dir}/rugd'
    
    
elif dataset == 'fishyscapes':
    diffusion_model_name = 'cityscapes_all/model-100.pt'
    dataset_path = '/home/sunsh16e/diffunc/data/fishyscapes/lost_found/images'
    save_path = '/home/sunsh16e/diffunc/experiment/fishyscapes'
    gen_save_path = save_path
    image_size = (256, 512)
    ori_image_size = (1024, 2048)

    ids = [2, 4, 7, 10, 27, 31, 46, 61, 74, 76, 84, 95]
    image_path_name = [f'lf{i}.png' for i in ids]
    image_paths= [os.path.join(dataset_path, e) for e in image_path_name]

elif dataset == 'rellis':
    model_path = f'{rellis_model_dir}/model-20.pt' #f'/home/sunsh16e/diffunc/models/rellis_full_id/model-30.pt'
    diffusion_model_name = None #'rugd_full_id_train/model-20.pt'
    if sp: 
        dataset_path = f'{data_dir}/RELLIS/ddpm_train_sets/full/images/ood'
    else:
        dataset_path = f'{data_dir}/RELLIS/full/images/ood' #/home/gridsan/groups/rrg/data/RELLIS/ddpm_train_sets/full
    gen_save_path = f'{dir}/rellis_20'
    image_size = (152, 240)
    ori_image_size = (1200, 1920)
    save_path = f'{dir}/rellis_20'

# conditioners = ['l2_p_7e4']
# upsample_models = ['dinov2']
print(dataset_path)
image_paths = sorted(get_image_paths(dataset_path))[::7]
print(image_paths[:2])
print('len', len(image_paths))
upsample_models = ['maskclip'] #, 'maskclip'] #'maskclip', #['dino16', 'dinov2', 'clip', 'maskclip', 'vit', 'resnet50']

for ddim_sampling_timesteps in ddim_sampling_timesteps_list:
    conditioners = [f'quant_ddim_{ddim_guidance_scale}_{ddim_sampling_timesteps}'] #, 'l2_p_2e4', 'l2_p_3e4', 'l2_p_5e4']#, 'l2_p_7e4', 'l2_p_1e5'] #, 'l2_p_3e5'] #[]'l2_q', 'l2_p', 'exp_linear_q', 'exp_linear_p']


    mask_generator = load_sam()
    for cond_type in conditioners:
        gen_img_path = os.path.join(gen_save_path, 'generated_images', cond_type)
        unc_save_path = os.path.join(save_path, 'unc_results', cond_type)
        fig_save_path = os.path.join(save_path, 'visualization', cond_type)

    folder_paths = [gen_img_path, unc_save_path, fig_save_path]
    for folder in folder_paths:
        if not os.path.exists(folder):
            os.makedirs(folder)  # Create the folder (and any intermediate ones)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
            
    pipe = Pipeline(diffusion_model_name = diffusion_model_name,
                    dataset_path = dataset_path, 
                    save_path = save_path,
                    image_size = image_size, 
                    conditioners = conditioners, 
                    upsample_models = upsample_models, 
                    mask_generator = mask_generator, 
                    verbose = verbose, 
                    report_num = report_num,
                    visualize_num=visualize_num,
                    overwrite = overwrite,
                    only_generate_img = only_generate_img,
                    square = square,
                    gen_save_path = gen_save_path, 
                    ori_image_size = ori_image_size,
                    model_path = model_path,
                    progress_bar = progress_bar,
                    ssim = ssim,
                    ddim = ddim,
                    ddim_sampling_timesteps = ddim_sampling_timesteps,
                    ddim_guidance_scale = ddim_guidance_scale,
                    )
    pipe.run(image_paths)


# from pipe import Pipeline
# from diffunc.all import load_sam, get_image_paths, extract_substring
# import os

# square = True
# only_generate_img = False
# overwrite = False # normally set as False
# verbose = False
# report_num = 10
# visualize_num = 20
# dataset = 'rellis'


# # load diffusion model
# if dataset == 'rugd':
#     diffusion_model_name = 'rugd_full_id_train/model-10.pt'
#     model_path = None
#     dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
#     gen_save_path = '/home/sunsh16e/diffunc/experiment/rugd'
#     image_size = (176, 216)
#     ori_image_size = (550, 688)
#     if square:
#         save_path = '/home/sunsh16e/diffunc/experiment/rugd_sqr_new'
#     else:
#         save_path = '/home/sunsh16e/diffunc/experiment/rugd'
    
    
# elif dataset == 'fishyscapes':
#     diffusion_model_name = 'cityscapes_all/model-100.pt'
#     model_path = None
#     dataset_path = '/home/sunsh16e/diffunc/data/fishyscapes/lost_found/images'
#     save_path = '/home/sunsh16e/diffunc/experiment/fishyscapes'
#     gen_save_path = save_path
#     image_size = (256, 512)
#     ori_image_size = (1024, 2048)

#     ids = [2, 4, 7, 10, 27, 31, 46, 61, 74, 76, 84, 95]
#     image_path_name = [f'lf{i}.png' for i in ids]
#     image_paths= [os.path.join(dataset_path, e) for e in image_path_name]

# elif dataset == 'rellis':
#     diffusion_model_name = 'rugd_full_id_train/model-10.pt'
#     model_path = '/home/sunsh16e/diffunc/models/rellis_full_id/model-30.pt'
#     dataset_path = '/home/sunsh16e/diffunc/data/RELLIS/full/images/ood'
#     gen_save_path = '/home/sunsh16e/diffunc/experiment/rellis_new'
#     image_size = (152, 240)
#     ori_image_size = (1200, 1920)
#     save_path = '/home/sunsh16e/diffunc/experiment/rellis_new'


# conditioners = ['l2_p_3e5'] #[]'l2_q', 'l2_p', 'exp_linear_q', 'exp_linear_p']
# mask_generator = load_sam()
# upsample_models = ['maskclip'] #['dino16', 'dinov2', 'clip', 'maskclip', 'vit', 'resnet50']



# pipe = Pipeline(diffusion_model_name = diffusion_model_name,
#                 dataset_path = dataset_path, 
#                 save_path = save_path,
#                 image_size = image_size, 
#                 conditioners = conditioners, 
#                 upsample_models = upsample_models, 
#                 mask_generator = mask_generator, 
#                 verbose = verbose, 
#                 report_num = report_num,
#                 visualize_num=visualize_num,
#                 overwrite = overwrite,
#                 only_generate_img = only_generate_img,
#                 square = square,
#                 gen_save_path = gen_save_path, 
#                 ori_image_size = ori_image_size,
#                 model_path = model_path
#                 )
# image_paths = sorted(get_image_paths(dataset_path))[::-1][500:]
# print(image_paths[:10])
# print('total num images', len(image_paths))
# # for rel_ood_img_path in image_paths:
# #     for cond_type in conditioners:
# #         pipe.visualize(rel_ood_img_path, cond_type)
# pipe.run(image_paths)


# #### check if all unc are here ###
# # import glob
# # upsampler_name = 'dinov2'
# # conditioners = ['l2']
# # for rel_ood_img_path in image_paths:
# #     for cond_type in conditioners:
# #         subname = extract_substring(rel_ood_img_path)
# #         unc_save_path = os.path.join(save_path, 'unc_results', cond_type, f'{subname}_{upsampler_name}.pt')
# #         base_save_path = os.path.join(save_path, 'unc_results', cond_type, f'{subname}_basics.pt')

# #         if not glob.glob(base_save_path):
# #             print('need', base_save_path)
# #         if not glob.glob(unc_save_path):
# #             print('need', base_save_path)


# # #### OnePipe test ########
# # model_name = 'rugd_full_id_train/model-10.pt'
# # model = Unet(
# #     dim = 64,
# #     dim_mults = (1, 2, 4, 8),
# #     flash_attn = True
# # )

# # diffusion = GaussianDiffusion(
# #     model,
# #     image_size = (176, 216),    # size of image (height, width) originally this was (128, 128)
# #     timesteps = 1000,           # number of steps
# #     sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
# #     objective = 'pred_v',       # note that default is different for guided_diffusion vs denoising_diffusion_pytorch
# # ).cuda()

# # # load model from checkpoint
# # model_path = download_from_cloud(model_name)
# # state_dict = torch.load(model_path)
# # diffusion.load_state_dict(state_dict['model'])

# # # display samples from model produced during training
# # sample_path = download_from_cloud(model_name.replace('model-', 'sample-').replace('.pt', '.png'))
# # IPyImage(sample_path, width=700)
# # cond_type = 'exp_linear_p'
# # # Set up the conditioner
# # cond_dic = {
# #     'l2_q': Conditioner(
# #             diffusion=diffusion,
# #             objectives=[
# #                 L2Objective(guidance_scale=1e5),
# #             ],
# #             apply_q_transform=True,
# #         ),
# #     'exp_linear_q': Conditioner(
# #             diffusion=diffusion,
# #             objectives=[
# #                 ExponentialPriorSPMObjective(guidance_scale=3e3),
# #             ],
# #             params=ScheduledParams(
# #                     width=(2.0,   0.1, 'linear'),
# #                 steepness=(0.1, 100.0, 'linear'),
# #                         λ=( 20,    20, 'fixed' ),
# #             ),
# #             apply_q_transform=True,
# #         ),
# #     'exp_linear_p': Conditioner(
# #             diffusion=diffusion,
# #             objectives=[
# #                 ExponentialPriorSPMObjective(guidance_scale=5e2),
# #             ],
# #             params=ScheduledParams(
# #                     width=(2.0,   0.1, 'linear'),
# #                 steepness=(0.1, 100.0, 'linear'),
# #                         λ=( 20,    20, 'fixed' ),
# #             ),
# #             apply_p_transform=True,
# #         )

# #     }
# # conditioner = cond_dic[cond_type]

# # upsampler = torch.hub.load("mhamilton723/FeatUp", 'clip', use_norm = True).to('cuda')

# # p = OneImgPipe(dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood', 
# #          save_path = '/home/sunsh16e/diffunc/experiment/rugd',
# #          rel_ood_img_path ='creek/creek_00051.png',
# #          image_size = (176, 216), cond_type = cond_type, conditioner = conditioner, upsampler = upsampler, upsampler_name = 'clip',
# #          verbose = True)
# # # p.get_generated_img(diffusion)
# # mask_generator = load_sam()
# # p.run_pipe(diffusion, mask_generator)
# # # print(torch.load(p.unc_save_path))
# # p.visualize(save = True)