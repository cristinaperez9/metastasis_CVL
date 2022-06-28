# Code to calculate the size of the metastases. In the literature, I found three ways of measuring the metastases size.

########################################################################################################################
# Cristina Almagro Perez, 2022. ETH University.
########################################################################################################################

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import measure
import scipy.io as sio

# SELECT ONE OF THE FOLLOWING:
option1 = True  # measuring the largest diameter in 3D
option2 = False  # project in the craniocaudal direction and measure the largest diameter
option3 = False  # project in the craniocaudal direction and measure the largest cross-sectional dimension (Used in MetNet paper)
#Code for all patient - analysing the size of metastases
pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/'
pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/'
names = ['Breast_nii', 'Melanoma_nii', 'NSCLC_nii', 'melanomapreopMRI_nii']
met_size = np.zeros((772, 1))
count = 0

for imyfiles in names:
    pth0 = os.path.join(pth00, imyfiles)  # 4 folders with the 4 types of cancers
    patients = os.listdir(pth0)
    for ipatients in patients:
        pth = os.path.join(pth0, ipatients)
        filename = os.path.join(pth, 'reg_mask_cropped_corrected.nii.gz')
        im = nib.load(filename)
        header = im.header
        pix_size = header.get_zooms()  # voxel size in mm
        factor = pix_size[0] * pix_size[1] * pix_size[2]  # mm^3
        im = nib.load(filename).get_fdata()
        labelled = measure.label(im)  # labelled components
        l = np.max(labelled)
        for x in range(1, l + 1):
            met = labelled == x
            met = np.multiply(met, 1)
        if option1:
            props = measure.regionprops(met)
            maj_ax_le = props[0].major_axis_length

        else:
            # Options 2 and 3
            # Common part to both
            met_proj = np.sum(met, axis=2)
            met_proj = met_proj > 0
            met_proj = np.multiply(met_proj, 1)

            if option2:
                props = measure.regionprops(met_proj)
                maj_ax_le = props[0].major_axis_length
            else: # option 3
                loc = np.nonzero(met_proj)
                xmin = np.min(loc[0])
                xmax = np.max(loc[0])
                ymin = np.min(loc[1])
                ymax =np.max(loc[1])
                width1 = xmax - xmin
                width2 = ymax - ymin
                if width1 > width2:
                    maj_ax_le = width1
                else:
                    maj_ax_le = width2
        print(maj_ax_le)
        met_size[count] = maj_ax_le
        count = count+1
sio.savemat('met_size_3D.mat', {'met_size': met_size})
