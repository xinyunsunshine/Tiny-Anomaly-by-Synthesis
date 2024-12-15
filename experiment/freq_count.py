import numpy as np
from PIL import Image as PilImage
from diffunc.all import get_image_paths
import torch
import os

CLASSES = ["void"] + \
    [
        'dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle',
        'container/generic-object', 'asphalt', 'gravel', 'building',
        'mulch', 'rock-bed', 'log', 'bicycle', 'person', 'fence', 'bush',
        'sign', 'rock', 'bridge', 'concrete', 'picnic-table'
    ]

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


train_paths = get_image_paths('/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/id/train', nosplit = False)
element_counts = dict()
c = 0
for rel_ood_img_path in train_paths[:1]:

    if c%100 == 0:
        print('progress:', c, rel_ood_img_path)
    c+=1
    src_label_file = f'/home/sunsh16e/diffunc/data/RUGD/RUGD_annotations/{rel_ood_img_path}'
    label_img_rgb = np.asarray(PilImage.open(src_label_file))
    label_img_idx = convert_label_img_rgb_to_id(label_img_rgb)
    
    unique_elements, counts = np.unique(label_img_idx, return_counts=True)
    for element, count in zip(unique_elements, counts):
        if element in element_counts:
            element_counts[element] += 1 #count
        else:
            element_counts[element] = 1 #count
print("Element counts:", element_counts)
new_element_counts = {CLASSES[key]: value for key, value in element_counts.items()}
print('New element counts', new_element_counts)

new_element_counts = dict(sorted(new_element_counts.items(), key=lambda item: item[1], reverse=True))
print('Sorted new element counts', new_element_counts)

total_pixels = len(train_paths) * label_img_idx.size
total_images =  len(train_paths)
new_element_counts = {3: 2314, 4: 2583, 7: 2377, 13: 1564, 18: 83, 0: 512, 5: 400, 11: 1818, 23: 12, 15: 1033, 20: 69, 1: 91, 21: 111, 19: 1073, 10: 19, 12: 6, 6: 153, 2: 87, 14: 3, 8: 19, 9: 22, 17: 2}
freq_dic = {key: round(value / total_images, 6) for key, value in new_element_counts.items()}
print('Sorted frequncy counts', dict(sorted(freq_dic.items())))


# ood_paths = get_image_paths('/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/images/ood', nosplit = False)
# label_save_dir = '/home/sunsh16e/diffunc/data/RUGD/ddpm_train_sets/full/labels/ood'
# c = 0
# for rel_ood_img_path in ood_paths:
#     if c%100 == 0:
#         print('progress:', c, rel_ood_img_path)
#     c+=1
#     src_label_file = f'/home/sunsh16e/diffunc/data/RUGD/RUGD_annotations/{rel_ood_img_path}'
#     label_img_rgb = np.asarray(PilImage.open(src_label_file))
#     label_img_idx = convert_label_img_rgb_to_id(label_img_rgb)

#     image = PilImage.fromarray(label_img_idx)
#     label_save_path = os.path.join(label_save_dir, rel_ood_img_path)
#     image.save(label_save_path)
    
  

