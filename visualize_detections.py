
# Code to visualize the predicted masks with or without the surrounding bounding boxes

#######################################################################################################################
# Cristina Almagro Perez, 2022. ETH University.
#######################################################################################################################

# Import necessary packages:
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pip
#pip.main(['install', 'opencv-python'])
import PIL
import cv2
from skimage import measure


###############################################################################################################
# Auxiliary functions
###############################################################################################################
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def get_coloured_mask(mask,gt=False):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    #r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[3]  #display mask in blue
    if gt:
        r[mask == 1], g[mask == 1], b[mask == 1] = colours[0]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def add_bb(df_bb,section,img,rgb_mask):
    rect_th = 2
    color = (255, 0, 0)

    # Pre-process image for cv2 format
    im = np.zeros((384, 384, 3))
    im[:, :, 0] = img
    im[:, :, 1] = img
    im[:, :, 2] = img
    im = convert(im, 0, 255, np.uint8)
    print("Section is...", section)
    for i in range(len(df_bb)):  # loop through the detected bounding boxes
        met_coords = df_bb["coords"].iloc[i]
        met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
        met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
        print(met_coords)
        # Check if metastases located in z range and plot
        if (section >= met_coords[4]) and (section <= met_coords[5]):
            start_point = np.floor(np.asarray(met_coords[0:2]))
            end_point = np.floor(np.asarray(met_coords[2:4]))
            start_point = np.flip(start_point.astype(int))
            end_point = np.flip(end_point.astype(int))
            start_point = (start_point.astype(int))
            end_point = (end_point.astype(int))
            cv2.rectangle(im, start_point, end_point, color, rect_th)
        plt.imshow(im, cmap='gray')


# SPECIFY THE FOLLOWING: 

# Define path for the location of the GT masks, predicted segmentation masks
pth_gt = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test0'

# Select paths that contain the predicted segmentation masks

# In original 3D Mask R-CNN implementation:
pth_pred = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/test_pred_seg/all_patients'
pth_bb_masks = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_hold_out_min_det_0.72/enlarged_5'


ipatient = 50  # Select patient to visualize
section = 189  # Select section to visualize
add_bb_square = False
patient_name = str(ipatient) + '.npy'

prob_map_thr = 0.4  # Threshold for the probability map output determined through a validation set
# 0.4 (Original Mask R-CNN implementation)
# 0.3 (Mask R-CNN with ResNet101 backbone)
# 0.5 (Mask R-CNN config 3 (see report))
# 0.1 (Retina U-Net)
# 0.4 (Detection U-Net)

#######################################################################################################################
# Visualize the specified patient and specified section
#######################################################################################################################

# Load GT mask
datafile = os.path.join(pth_gt, patient_name)
gt_data = np.load(datafile)
gt_mask = gt_data[1, :, :, :]
# Find slices that contain gt metastases
print("The slices of this patient that contain GT metastases are...")
loc = np.sum(np.sum(gt_mask, 0), 0)
loc = np.nonzero(loc)
print(loc)

# Load predicted mask
datafile = os.path.join(pth_pred, patient_name)
pred_mask = np.squeeze(np.load(datafile))  # probability map
pred_mask = np.multiply(pred_mask > prob_map_thr, 1)
# Load bounding box mask
datafile = os.path.join(pth_bb_masks, patient_name)
bb_mask = np.load(datafile)
pred_mask = np.multiply(bb_mask, pred_mask)
# Find slices pred metastases
loc = np.sum(np.sum(pred_mask, 0), 0)
print("The slices of the predicted segmentation mask of this patient that contain metastases are...")
loc = np.nonzero(loc)
print(loc)


# Show only MR image
im = gt_data[0, :, :, :]
plt.imshow(im[:, :, section], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

# Show GT metastasis overlaid in the MR image
im = gt_data[0, :, :, :]
met_mask_coloured = get_coloured_mask(gt_mask[:, :, section], gt=True)
plt.imshow(im[:, :, section], cmap='gray')
plt.imshow(met_mask_coloured, alpha=0.3)
plt.xticks([])
plt.yticks([])
plt.show()

# Show prediction overlaid in the MR image
pred_mask_coloured = get_coloured_mask(pred_mask[:, :, section], gt=False)
plt.imshow(im[:, :, section], cmap='gray')
plt.imshow(pred_mask_coloured, alpha=0.3)
plt.xticks([])
plt.yticks([])
plt.show()

