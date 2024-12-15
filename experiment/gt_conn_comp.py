import numpy as np
from PIL import Image as PilImage
from diffunc.all import get_image_paths
import torch
import os, glob
from scipy.ndimage import label, find_objects

def relabel_connected_components(arr, threshold=500):
    unique_values = np.unique(arr)
    structure = np.ones((3, 3), dtype=np.uint8)  # You can change this structure if needed

    relabeled_array = np.zeros_like(arr, dtype=np.int32)
    current_label = 1

    for value in unique_values:
        if value == 0: # skip void
            continue  # Skip the background or zero value if you have one

        # Create a binary mask for the current value
        mask = (arr == value)
        
        # Label the connected components in the mask
        labeled_mask, num_features = label(mask, structure=structure)
        
        # Get the slices for each component
        component_slices = find_objects(labeled_mask)
        
        for i in range(num_features):
            component = (labeled_mask == (i + 1))
            component_size = np.sum(component)
            
            if component_size >= threshold:
                relabeled_array[component] = current_label
                current_label += 1

    return relabeled_array

def relabel_small(arr, image_ar, pixel_ar, threshold=500):
    """_summary_

    Args:
        arr (np array): label 
        image_ar (np array): image wise freq 
        pixel_ar (np array): pixel wise freq
        threshold (int, optional): min component size in pixel space. Defaults to 500.

    Returns:
        image_cp, pixel_cp (tensor): new image and pixel wise freq tensor with small components eliminated
    """
    unique_values = np.unique(arr)
    structure = np.ones((3, 3), dtype=np.uint8)  # You can change this structure if needed

    image_cp = np.copy(image_ar)
    pixel_cp = np.copy(pixel_ar)

    for value in unique_values:
        if value == 0: # skip void
            continue  # Skip the background or zero value if you have one

        # Create a binary mask for the current value
        mask = (arr == value)
        
        # Label the connected components in the mask
        labeled_mask, num_features = label(mask, structure=structure)
        
        # Get the slices for each component
        component_slices = find_objects(labeled_mask)
        
        for i in range(num_features):
            component = (labeled_mask == (i + 1))
            component_size = np.sum(component)
            
            if component_size < threshold:
                image_cp[component] = 2
                pixel_cp[component] = 2

    return torch.tensor(image_cp), torch.tensor(pixel_cp)

rugd = False #rellis
if rugd:
    ood_paths = get_image_paths('/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood')
    label_save_dir = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/labels/ood'
else:
    ood_paths = ['00002/frame000518-1581797202_209.jpg'] #get_image_paths('/home/sunsh16e/diffunc/data/RELLIS/full/images/ood')
    label_save_dir =  '/home/sunsh16e/diffunc/data/RELLIS/full/labels/ood/'
c = 0
print('num images', len(label_save_dir))
threshold = 1200 * 1920 * 1.5e-3
for rel_ood_img_path in ood_paths[:1]:
    if c%100 == 0:  
        print('progress:', c, rel_ood_img_path)
    c+=1
    label_path = os.path.join(label_save_dir, rel_ood_img_path[:-4])+ '.pt'
    print(label_path)
    
    dic = torch.load(label_path)
    if 'new_label' not in dic:
        label_ar = dic['label'] #.numpy()
        image_ar = dic['image'].numpy()
        pixel_ar = dic['pixel'].numpy()
        new_image, new_pixel = relabel_small(label_ar, image_ar, pixel_ar, threshold = threshold)
        dic = dic | {'new_image': new_image, 'new_pixel': new_pixel}
        torch.save(dic, label_path)

    # if 'con_label' not in dic:
    #     label_ar = dic['label'] #.numpy()
    #     con_label = relabel_connected_components(label_ar)
    #     dic = dic | {'con_label': con_label}
    #     torch.save(dic, label_path)

    
  

