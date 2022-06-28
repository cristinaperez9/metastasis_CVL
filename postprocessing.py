

#######################################################################################################################
# Cristina Almagro Perez, 2022. ETH University
#######################################################################################################################

# Code to create and save mask of the predicted bounding boxes (the model only outputs the box coordinates)

# Import necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################################################################################################
#  Specify the following:
#####################################################################################################################
thrf = 0.79  # Optimal prediction score that maximized the F1 score in the validation set; The values for the models
# tested in the project were: 0.72 (original Mask R-CNN implementation), 0.69 (Mask R-CNN with ResNet101),
# 0.79 (Mask R-CNN with config 3, (see report), 0.79 (Retina U-Net), and 0.70 (Detection U-Net)
# #0.72 #0.69
thr = 0.5  # thr: minimum prediction score that was required during inference (specified in config.py file)
n_patients = 54
enlarge_box = True  # Leave some pixels outside the bounding box that can be segmented
enlarge_val = 5  # Number of pixels at each side of the tight bounding box surrounding a metastasis

pth_gt = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test0'

# PATHS THAT WERE USED WITH THE MODELS TESTED IN THE PROJECT:
pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/'
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_1_ResNet101/'
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_0_param_300/'
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_3_retina_unet/'
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_4_detection_unet/'

# Note: for efficiency, it is better to save in scratch_net than /usr/bmicdatasts/
if enlarge_box:
    outpth = os.path.join(pth, 'results_hold_out_min_det_' + str(thrf), 'enlarged_5')
    if not os.path.exists(outpth):
        os.makedirs(outpth)

else:
    outpth = os.path.join(pth, 'results_hold_out_min_det_' + str(thr))
if not os.path.exists(outpth):
    os.makedirs(outpth)

# Load file containing the bounding box coordinates of all patients
nm = 'results_hold_out' + '.csv'
datafile = os.path.join(pth, nm)
file_bb = pd.read_csv(datafile)
file_bb = file_bb[file_bb["score"] > thrf]

# Load patient info
datafile = os.path.join(pth_gt, 'info_df.pickle')
info_df = pd.read_pickle(datafile)
for ipatient in range(1, n_patients + 1):

    print("Postprocessing patient... ", ipatient)
    patient_name = str(ipatient) + '.npy'

    # Load GT mask
    datafile = os.path.join(pth_gt, patient_name)
    gt_mask = np.load(datafile)
    gt_mask = gt_mask[1, :, :, :]
    output = np.zeros(gt_mask.shape)

    # Load file with coordinates
    df_bb = file_bb[file_bb["patientID"] == ipatient]
    print(df_bb)
    if df_bb.empty:
        print("Creating empty mask")
    else:
        for kk in range(len(df_bb)):
            met_coords = df_bb["coords"].iloc[kk]
            met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
            met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
            if enlarge_box:
                output[met_coords[0]-enlarge_val:met_coords[2]+enlarge_val, met_coords[1]-enlarge_val:
                met_coords[3] + enlarge_val, met_coords[4]-enlarge_val:met_coords[5]+enlarge_val] = 1
            else:
                output[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
    output_datafile = os.path.join(outpth, patient_name)
    np.save(output_datafile, output)

