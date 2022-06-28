
#######################################################################################################################
# Cristina Almagro Perez, 2022. ETH Zurich.
#######################################################################################################################

# Code to preprocess the data in format .nii to be able to run in within the Medical Detection Framework.

'''
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

# Import necessary packages
import os
from natsort import natsorted
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import matplotlib.pyplot as plt
import configs
cf = configs.configs()


#######################################################################################################################
# Auxiliary functions
#######################################################################################################################

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in natsorted(os.listdir(exp_dir)) if 'meta_info' in f]
    df = pd.DataFrame(columns=['path_image', 'pid', 'patient_name'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print("aggregated meta info to df with length", len(df))

####################################################################################################################
# MAIN
####################################################################################################################


dataset_type = 'test/'  # choose between train or test
raw_data_dir_img = os.path.join(cf.raw_data_dir, dataset_type, 'images/')
raw_data_dir_mask = os.path.join(cf.raw_data_dir, dataset_type, 'masks/')

outpth = os.path.join(cf.raw_data_dir, 'preprocessed/', dataset_type)

myfiles = sorted(os.listdir(raw_data_dir_img))
#myfiles = myfiles[1:]  # The first image is corrupted (include this only in the train case!)
count = 0

for imyfiles in myfiles:  # Loop through all patients
    count += 1

    # Data preprocessing
    patient_name = imyfiles[0:len(myfiles)-8]
    datafile = os.path.join(raw_data_dir_img, imyfiles)
    img_arr = nib.load(datafile).get_fdata()  # Load image
    datafile = os.path.join(raw_data_dir_mask, imyfiles)
    mask_arr = nib.load(datafile).get_fdata()  # Labelled mask loaded
    mask_arr = mask_arr > 0
    mask_arr = np.multiply(mask_arr, 1)
    print('processing image of patient... {}'.format(patient_name))

    # STEP 1: Normalize images between 0 and 1
    image = convert(img_arr, 0, 255, np.uint8)  # Image normalized between 0 - 255
    img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))  # Image normalized between 0 - 1

    # STEP 2: Extend the z-dimension so instead of 270 is 272 (this dimension has to be divisible by 2 three times)
    img_arr_2 = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]+2))
    img_arr_2[:, :, 1: img_arr.shape[2]+1] = img_arr  # bulk of the image
    img_arr_2[:, :, 0] = img_arr[:, :, 0]  # Add one bottom slice that is the first slice
    img_arr_2[:, :, img_arr.shape[2]+1] = img_arr[:, :, img_arr.shape[2]-1]  # Add one top slice that is the last slice
    img_arr_2 = img_arr_2.astype(np.float64)

    mask_arr_2 = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2] + 2))
    mask_arr_2[:, :, 1: img_arr.shape[2]+1] = mask_arr  # bulk of the mask
    mask_arr_2[:, :, 0] = mask_arr[:, :, 0]  # Add one bottom slice that is the first slice
    mask_arr_2[:, :, img_arr.shape[2] + 1] = mask_arr[:, :, img_arr.shape[2] - 1]  # Add one top slice that is the last slice
    mask_arr_2 = mask_arr_2.astype(np.float64)

    # Concatenate image & mask
    out = np.concatenate((img_arr_2[None], mask_arr_2[None]))  # Indexing with none adds a dimension to your array
    outpth_image = os.path.join(outpth, '{}.npy'.format(count))
    np.save(outpth_image, out)  # Save concatenated image & mask as npy format

    # Save meta-data for each of the images
    with open(os.path.join(outpth, 'meta_info_{}.pickle'.format(count)), 'wb') as handle:
        pickle.dump([outpth_image, str(count), patient_name], handle)  # The second column will be the patient_ID & the third column the patient_name

# Aggregate meta-data information in folder
aggregate_meta_info(outpth)
