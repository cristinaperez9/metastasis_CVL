## Code for analyzing the data statistics of metastases:
# - Number of metastases per patient
# - Number of metastases in each of the primary cancer types
# - Analysis of the metastases size

#--------------------------------------------------------------------------
# Code written by Cristina Almagro-Perez,2022, ETH University (Zurich).
#--------------------------------------------------------------------------
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import measure
import scipy.io as sio

###########################################################################################
# SPECIFY THE FOLLOWING:
primary_cancer = True   #Count how many patients there are of each cancer
met_per_patient = True  #Analyze the number of metastases per patient
met_size = True  #Analyze the frequency of metastases based on size
###########################################################################################


if primary_cancer:
    #Count how many patients there are of each cancer - corrected version to avoid .DS_Store files
    pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/'
    freq = np.zeros((1, 4))
    myfiles = os.listdir(pth00)
    for imyfiles in myfiles:
        pth0 = os.path.join(pth00, imyfiles)  # 4 folders with the 4 types of cancers
        patients = os.listdir(pth0)
        print(len(patients))

if met_per_patient:
    ## UNCOMMENT THE FOLLOWING CODE TO TEST ALGORITHM FOR ONE PATIENT ##
    # Count how many metastasis we have per patient
    #Code for one patient
    # pth=r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/Melanoma_nii/000012-20'
    # filename = os.path.join(pth, 'reg_mask_cropped_corrected.nii.gz')
    # im = nib.load(filename).get_fdata()
    # num=measure.label(im)
    # print((np.max(num)))

    # Code for all patients
    pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/'
    myfiles = os.listdir(pth00)
    data = np.zeros((273, 1))  #total of 273 patients
    count = 0
    for imyfiles in myfiles:
        pth0 = os.path.join(pth00, imyfiles)  # 4 folders with the 4 types of cancers
        patients = os.listdir(pth0)
        for ipatients in patients:
            pth = os.path.join(pth0, ipatients)
            filename = os.path.join(pth, 'reg_mask_cropped_corrected.nii.gz')
            img = nib.load(filename).get_fdata()
            num = measure.label(img)
            data[count] = np.max(num)
            count = count + 1
    print(count)
    sio.savemat('data.mat', {'data': data})

if met_size:
    ## UNCOMMENT THE FOLLOWING CODE TO TEST ALGORITHM FOR ONE PATIENT ##
    # Frequency of metastasis based on size
    # Code for one patient - modified code considering the projection onto the craniocaudal direction
    # pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/Melanoma_nii/000001-10'
    # filename = os.path.join(pth, 'reg_mask_cropped_corrected.nii.gz')
    # #pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_segmentation/Processed/Registered/Breast_nii/000026-9'
    # #filename = os.path.join(pth, 'reg_img_cropped.nii.gz')
    # im = nib.load(filename)
    # header = im.header
    # pix_size = header.get_zooms()  #mm
    # factor = pix_size[0]*pix_size[1]*pix_size[2] #mm^3
    # print(factor)
    # im = nib.load(filename).get_fdata()
    # labelled = measure.label(im) #labelled components
    # l=np.max((labelled))
    # for x in range(1,l+1):
    #     met=labelled==x
    #     met = np.multiply(met, 1)
    #     met_proj = np.sum(met,axis = 2)
    #     met_proj = met_proj > 0
    #     met_proj = np.multiply(met_proj, 1)
    #     print(met_proj.shape)
    #     props = measure.regionprops(met_proj)
    #     maj_ax_le = props[0].major_axis_length
    #     print(maj_ax_le)
    #
    #     props = measure.regionprops(met)
    #     maj_ax_le = props[0].major_axis_length
    #     print(maj_ax_le)
    # plt.savefig("met_proj.png")

    #Code for all patient - analysing the size of metastases
    pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/'
    myfiles = os.listdir(pth00)
    met_size = np.zeros((772, 1))  # 772 metastases in this dataset
    count = 0
    for imyfiles in myfiles:
        pth0 = os.path.join(pth00, imyfiles)  # 4 folders with the 4 types of cancers
        patients = os.listdir(pth0)
        for ipatients in patients:
            pth = os.path.join(pth0, ipatients)
            filename = os.path.join(pth, 'reg_mask_cropped_corrected.nii.gz')
            im = nib.load(filename)
            header = im.header
            pix_size = header.get_zooms()  # mm
            factor = pix_size[0] * pix_size[1] * pix_size[2]  # mm^3
            im = nib.load(filename).get_fdata()
            labelled = measure.label(im)  # labelled components
            l = np.max(labelled)
            for x in range(1, l + 1):
                met = labelled == x
                met = np.multiply(met, 1)
                met_proj = np.sum(met, axis=2)
                met_proj = met_proj > 0
                met_proj = np.multiply(met_proj, 1)
                print(met_proj.shape)
                props = measure.regionprops(met_proj)
                maj_ax_le = props[0].major_axis_length
                met_size[count] = maj_ax_le
                count = count+1
    sio.savemat('met_size.mat', {'met_size': met_size})
