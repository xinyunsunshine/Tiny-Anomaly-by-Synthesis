import numpy as np
from PIL import Image as PilImage
from diffunc.all import get_image_paths
import torch
import os, glob
from scipy.ndimage import label, find_objects

CLASSES = ["void"] + \
    [
        'dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle',
        'container/generic-object', 'asphalt', 'gravel', 'building',
        'mulch', 'rock-bed', 'log', 'bicycle', 'person', 'fence', 'bush',
        'sign', 'rock', 'bridge', 'concrete', 'picnic-table'
    ]
print(len(CLASSES))

OOD_CLASSES = \
[
    'vehicle', 'container/generic-object', 'building', 'bicycle',
    'person', 'bridge', 'picnic-table'
]
OOD_CLASS_INDICES = [CLASSES.index(cls_name) for cls_name in OOD_CLASSES]
print('ood class indices', OOD_CLASS_INDICES)

PALETTE = [[0, 0, 0]] + \
    [
        [108, 64, 20], [255, 229, 204], [0, 102, 0], [0, 255, 0],
        [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0],
        [255, 0, 127], [64, 64, 64], [255, 128, 0], [255, 0, 0],
        [153, 76, 0], [102, 102, 0], [102, 0, 0], [0, 255, 128],
        [204, 153, 255], [102, 0, 204], [255, 153, 204], [0, 102, 102],
        [153, 204, 255], [102, 255, 255], [101, 101, 11], [114, 85, 47]
    ]

def convert_label_img_rgb_to_id(rgb_img):
    # set default label id to 255
    id_img = 255 * np.ones(shape=rgb_img.shape[:2], dtype=np.uint8)
    
    # match pixels to ids via their RGB values
    for id, rgb in enumerate(PALETTE):
        id_img[(rgb_img == rgb).all(axis=2)] = id
    
    # sanity check to ensure that each pixel has been matched
    assert (id_img < len(CLASSES)).all()

    return id_img

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


freq_dict = {0: 0.000165, 1: 0.001015, 2: 0.001802, 3: 0.237834, 4: 0.433998, 5: 0.000598, 6: 0.000775, 7: 0.057078, 8: 1e-06, 9: 1e-06, 10: 2.8e-05, 11: 0.102058, 12: 0.0, 13: 0.121569, 14: 0.000158, 15: 0.005251, 16: 0, 17: 0.0, 18: 0.004682, 19: 0.031277, 20: 2.7e-05, 21: 0.000902, 22:0, 23: 0.00078, 24: 0}
img_freq_dict = {0: 0.198142, 1: 0.035217, 2: 0.033669, 3: 0.895511, 4: 0.999613, 5: 0.154799, 6: 0.059211, 7: 0.919892, 8: 0.007353, 9: 0.008514, 10: 0.007353, 11: 0.70356, 12: 0.002322, 13: 0.605263, 14: 0.001161, 15: 0.399768, 16: 0, 17: 0.000774, 18: 0.032121, 19: 0.415248, 20: 0.026703, 21: 0.042957, 22:0, 23: 0.004644, 24: 0}

vectorized_replace = np.vectorize(freq_dict.get)
img_vectorized_replace = np.vectorize(img_freq_dict.get)
ood_paths = get_image_paths('/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood')
label_save_dir = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/labels/ood'
c = 0
print('num images', len(label_save_dir))
for rel_ood_img_path in ood_paths:
    if c%100 == 0:  
        print('progress:', c, rel_ood_img_path)
    c+=1
    label_path = os.path.join(label_save_dir, rel_ood_img_path[:-4])+ '.pt'
    if glob.glob(label_path):
        continue
    src_label_file = f'/home/sunsh16e/diffunc/data/RUGD/RUGD_annotations/{rel_ood_img_path}'
    label_img_rgb = np.asarray(PilImage.open(src_label_file))
    label_img_idx = convert_label_img_rgb_to_id(label_img_rgb)
    pixel_label = vectorized_replace(label_img_idx)
    img_label = img_vectorized_replace(label_img_idx)
    mask = (label_img_idx == 0)

    label_ar = label_img_idx #.numpy()
    con_label = relabel_connected_components(label_ar)



    t = {'label': label_img_idx, 'pixel': torch.tensor(pixel_label), 'image':torch.tensor(img_label), 'con_label': con_label}
    print('saved at', label_path)
    torch.save(t, label_path)

