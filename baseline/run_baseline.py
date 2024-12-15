from knn import KNN
from gmm import GMM
from diffunc.all import load_input_img,  get_image_paths, extract_substring
import os
import glob
import torch

gen = False
baseline = 'gmm_20' #'gmm_200' #'knn' #'gmm'
dataset ='rellis'
if dataset == 'rugd':
    test_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood' #'/home/sunsh16e/diffunc/experiment/rugd/generated_images' #'/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
    train_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/id/train'
    save_path = '/home/sunsh16e/diffunc/baseline/data'
    parent_dir = '/home/sunsh16e/diffunc/baseline/data'
elif dataset == 'rellis':
    test_dataset_path = '/home/sunsh16e/diffunc/data/RELLLIS/full/images/ood' #'/home/sunsh16e/diffunc/experiment/rugd/generated_images' #'/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
    train_dataset_path = '/home/sunsh16e/diffunc/data/RELLLIS/full/images/id'
    save_path = '/home/sunsh16e/diffunc/baseline/rellis_data'
    parent_dir = '/home/sunsh16e/diffunc/baseline/rellis_data'
print('parent_dir', parent_dir)
train_bank_size = 10000
upsample_models = ['dinov2'] # ['dino16', 'clip', 'maskclip', 'vit', 'resnet50'] 
highres = True
n_components = 20 #200
train_max_iter=50
train_n_init = 1

for upsample_model in upsample_models:
    print(upsample_model)
    if baseline == 'knn':
        model = KNN(max_size= train_bank_size, upsample_model = upsample_model, highres = highres, parent_dir = parent_dir)
    elif 'gmm' in baseline:
        model = GMM(train_bank_size= train_bank_size, n_components = n_components,
            upsample_model = upsample_model, highres = highres,parent_dir = parent_dir,
            train_max_iter = train_max_iter, train_n_init = train_n_init, baseline = baseline)

    print('train')
    train_image_paths = sorted(get_image_paths(train_dataset_path))
    model.train(train_dataset_path, train_image_paths)

    # inference
    print('eval')
    if gen:
        test_image_paths = sorted(get_image_paths(test_dataset_path, folders = ['l2_p']))
        print(test_image_paths[:5])
    else:
        test_image_paths = sorted(get_image_paths(test_dataset_path))
    count = 0
    for rel_ood_img_path in test_image_paths:
        if gen:
            unc_save_path = os.path.join(parent_dir,  baseline, 'gen_unc', upsample_model, f'{extract_substring(rel_ood_img_path)}.pt')
        else:
            unc_save_path = os.path.join(parent_dir,  baseline, 'unc', upsample_model, f'{extract_substring(rel_ood_img_path)}.pt')
        # print(unc_save_path)
        if glob.glob(unc_save_path):
            continue
        ori_img_path = os.path.join(test_dataset_path, rel_ood_img_path)
        input = load_input_img(ori_img_path, no_cuda=False)
        unc = model.forward(input)
        torch.save(unc, unc_save_path)
        if count%10 ==0:
            print(count, rel_ood_img_path)
        count+=1