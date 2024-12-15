from knn import KNN
from diffunc.all import load_input_img,  get_image_paths, extract_substring
import os
import torch



test_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood'
train_dataset_path = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/id/train'
save_path = '/home/sunsh16e/diffunc/baseline/data'
max_size = 100000
upsample_model = 'dinov2'
highres = False
knn = KNN(max_size= max_size, upsample_model = upsample_model, highres = highres)

# training
train_image_paths = sorted(get_image_paths(train_dataset_path))
for rel_ood_img_path in train_image_paths[:1]:
    ori_img_path = os.path.join(train_dataset_path, rel_ood_img_path)
    input = load_input_img(ori_img_path, no_cuda=False)
    knn.train(input)
bank_save_path = os.path.join(save_path, 'vector_bank', f'{upsample_model}.pt')
torch.save(knn.bank_builder.bank, bank_save_path)
knn.bank_builder.bank.shape

# inference
test_dataset_paths = sorted(get_image_paths(test_dataset_path))
for rel_ood_img_path in test_dataset_paths[:1]:
    ori_img_path = os.path.join(test_dataset_path, rel_ood_img_path)
    input = load_input_img(ori_img_path, no_cuda=False)
    unc = knn.forward(input)
    unc_save_path = os.path.join(save_path, 'knn', 'unc',upsample_model, f'{extract_substring(rel_ood_img_path)}.pt')
    torch.save(unc, unc_save_path)
