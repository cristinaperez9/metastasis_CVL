# Code to evaluate the detection of brain metastases.

########################################################################################################################
# Cristina Almagro Perez, 2022. ETH University
########################################################################################################################



# Import necessary packages
import os
import pandas as pd
import pickle
import numpy as np
import sys
from skimage import measure
import scipy.io as sio
ROOT_DIR = os.path.abspath(r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/')
sys.path.append(ROOT_DIR)  # To find local version of the library
import operator

##################################################################################################################
# Auxiliary functions
##################################################################################################################

def find_ind(gt_mask_labelled, bb_met_image):
    iou_max = 0
    ind = 0
    for i in range(1, np.max(gt_mask_labelled) + 1):
        met = np.multiply(gt_mask_labelled == i, 1)
        iou_value = compute_iou_patient(met, bb_met_image)
        if iou_value > iou_max:
            iou_max = iou_value
            ind = i
    return ind


def obtain_size_met(met):

    met_proj = np.sum(met, axis=2)
    met_proj = met_proj > 0
    met_proj = np.multiply(met_proj, 1)
    props = measure.regionprops(met_proj)
    maj_ax_le = props[0].major_axis_length

    return maj_ax_le


def compute_dice_patient(gt, pred):

    n = 2 * np.sum(np.multiply(gt, pred).flatten())
    d = np.sum(gt.flatten()) + np.sum(pred.flatten())
    dice_value = n / d

    return dice_value


def compute_iou_patient(gt, pred):

    n = np.sum(np.multiply(gt, pred).flatten())
    d = gt + pred
    d = np.sum(np.multiply(d > 0, 1))
    iou_value = n / d

    return iou_value

########################################################################################################################
# SPECIFY THE FOLLOWING:


thr = 0.5   # thr: minimum prediction score that was required during inference (specified in config.py file)
thrf = 0.79  # Optimal prediction score that maximized the F1 score in the validation set; The values for the models
# tested in the project were: 0.72 (original Mask R-CNN implementation), 0.69 (Mask R-CNN with ResNet101),
# 0.79 (Mask R-CNN with config 3, (see report), 0.79 (Retina U-Net), and 0.70 (Detection U-Net)
# #0.72 #0.69
match_iou = 0.1  # IoU between prediction and GT to consider a metastasis True Positive (specified in config.py file)
save_tp_bb = False  # Model outputs the bounding box coordinates of the metastases, set to True to save masks with the
# bounding boxes
name_fold = 3   # Either 0, 1, 2, 3, or 4.

# PLEASE MODIFY THE FOLLOWING ACCORDING TO YOUR PATHS:
# Path where ground truth masks are stored
pth_gt = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test0'
# Path of predicted segmentation masks
pth_pred = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/test_pred_seg/all_patients/'
# Path of the predicted metastases' bounding boxes
pth_bb_masks = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_hold_out_min_det_0.72/enlarged_5'
# Path to save the masks with the bounding boxes of true positive metastases
outpth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_hold_out_min_det_0.9/only_tp_bb'

# PATHS THAT WERE USED FOR THE DIFFERENT MODELS TESTED:

# Variables for ResNet101 (my_3D_dataset_modify_architectures/fold1)
pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_1_ResNet101/'
#outpth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/ResNet101/results_hold_out_min_det_0.69/only_tp_bb/'

# Variables for Conf 2 (Configuration 4, parameters = 300)
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_0_param_300/'
#outpth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/param300/results_hold_out_min_det_0.79/only_tp_bb/'
# Variables for Retina_Unet (my_3D_dataset_modify_architectures/fold3)
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_3_retina_unet/'
#outpth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/retina_unet/results_hold_out_min_det_0.79/only_tp_bb/'

# Variables for Detection_Unet (my_3D_dataset_modify_architectures/fold3)
#pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_4_detection_unet/'
#outpth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/retina_unet/results_hold_out_min_det_0.79/only_tp_bb/'

########################################################################################################################

# Load files containing the final predictions per patient & concatenate them (test1, test2 & test3). This split was
# required due to memory limitations in the cluster

nm = 'results_hold_out' + '.csv'
if os.path.exists(os.path.join(pth, nm)):
    file_bb = pd.read_csv(os.path.join(pth, nm))
else:
    print("Creating file results_hold_out")
    datafile1 = os.path.join(pth, 'results_test1/', nm)
    file1 = pd.read_csv(datafile1)
    datafile2 = os.path.join(pth, 'results_test2/', nm)
    file2 = pd.read_csv(datafile2)
    datafile3 = os.path.join(pth, 'results_test3/', nm)
    file3 = pd.read_csv(datafile3)
    frames = [file1, file2, file3]
    file_bb = pd.concat(frames)
    print(file_bb)
    file_bb.to_csv(os.path.join(pth, nm), index=False)

# Load file containing the final predictions saying the type of prediction: TP, FP or FN:

nm = str(name_fold) + '_test_df.pickle'

if os.path.exists(os.path.join(pth, nm)):
    file_det_type = pd.read_pickle(os.path.join(pth, nm))
else:
    print("Creating file test_df")
    datafile1 = os.path.join(pth, 'results_test1/', nm)
    print(datafile1)
    file1 = pd.read_pickle(datafile1)
    datafile2 = os.path.join(pth, 'results_test2/', nm)
    file2 = pd.read_pickle(datafile2)
    datafile3 = os.path.join(pth, 'results_test3/', nm)
    file3 = pd.read_pickle(datafile3)
    frames = [file1, file2, file3]
    file_det_type = pd.concat(frames)
    print(file_det_type)
    with open(os.path.join(pth, nm), 'wb') as handle:
        pickle.dump(file_det_type, handle)

########################################################################################################################
# Set to True the analyses you desire to perform
########################################################################################################################
determine_thrf = True  # Determine the optimal minimum confidence score to consider a metastasis TP
evaluate_detection = True  # Obtain precision, accuracy, recall and F1 score. It requires to have specified thrf above.
evaluate_detection_size = True  # Evaluate detection metrics on small and large metastases.
volume_graph = True  # Obtain predicted and GT metastasis volume for Bland-Altman plot
missed_and_found = True  # Identify from the GT metastases the TP and FN as a function of the metastasis volume
fp_per_patient = True  # Obtain the number of false positives per patient
whisker_plot = True  # for TP metatases save the volume and DICE score to analyze DICE value as a function of met size
per_patient = True  # obtain average precision and recall per patient

if evaluate_detection:

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    count_TP = 0
    count_FP = 0
    count_FN = 0

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]
    for ipatient in range(1, n_patients + 1):

        image_bb_tp = np.zeros((384, 384, 272))

        # Initialize values
        TP = 0
        FN = 0
        FP = 0

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) == np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]
                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]

                if det_type == 'det_tp':
                    TP = TP + 1
                    image_bb_tp[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.1, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :]  # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")
                            TP = TP + 1
                            image_bb_tp[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                        else:
                            FP = FP + 1
                    else:
                        FP = FP + 1

        # Calculate FN
        df_det_type_met = df_det_type[df_det_type["class_label"] == 1]  # Find predictionsIDs that are metastases
        predictionsIDs = np.array(df_det_type_met.T.columns) - np.array(df_det_type.T.columns)[0]
        gt_num_met = len(predictionsIDs)

        FN = gt_num_met - TP

        count_TP = count_TP + TP
        count_FP = count_FP + FP
        count_FN = count_FN + FN

        print("The total number of TP in this patient is...", TP)
        print("The total number of FP in this patient is...", FP)
        print("The total number of FN in this patient is...", FN)

        # Save image with TP bounding boxes
        if save_tp_bb:
            if not os.path.exists(outpth):
                os.makedirs(outpth)
            patient_name = str(ipatient) + '.npy'
            output_datafile = os.path.join(outpth, patient_name)
            np.save(output_datafile, image_bb_tp)

    print("The total number of TP is...",  count_TP)
    print("The total number of FP is...", count_FP)
    print("The total number of FN is...", count_FN)

    # Calculate precision and recall
    recall = count_TP/(count_TP + count_FN)
    precision = count_TP/(count_TP + count_FP)

    print("The recall is...", recall)
    print("The precision is...", precision)

    # Calculate F1 & accuracy
    F1 = 2 * precision * recall / (precision + recall)
    print("The F1 score is...", F1)

    acc = count_TP / (count_TP + count_FP + count_FN)
    print("The accuracy is...", acc)


#############################################################################################################
# Evaluate detection considering dimension of the metastasis (>6mm or <6mm)
#############################################################################################################

if evaluate_detection_size:

    prob_map_thr = 0.4  # Threshold for the probability map output determined through a validation set
    # 0.4 (Original Mask R-CNN implementation)
    # 0.3 (Mask R-CNN with ResNet101 backbone)
    # 0.5 (Mask R-CNN config 3 (see report))
    # 0.1 (Retina U-Net)
    # 0.4 (Detection U-Net)

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    count_TP_large = 0  # Large refers to metastases greater than 6mm
    count_FP_large = 0
    count_FN_large = 0
    count_TP_small = 0  # Large refers to metastases smaller than 6mm
    count_FP_small = 0
    count_FN_small = 0

    for ipatient in range(1, n_patients + 1):

        # Initialize values
        TP_large = 0
        FN_large = 0
        FP_large = 0

        TP_small = 0
        FN_small = 0
        FP_small = 0

        print("Evaluating patient... ", ipatient)
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Load GT mask of patient
        patient_name = str(ipatient) + '.npy'
        datafile = os.path.join(pth_gt, patient_name)
        gt_data = np.load(datafile)
        gt_mask = gt_data[1, :, :, :]
        gt_mask_labelled = measure.label(gt_mask)

        # keep TP indices
        keep_tp_ind = [] #list

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                met_bb = df_bb.iloc[met_num, :]
                met_dt = df_det_type.iloc[met_bb["predictionID"], :]
                det_type = met_dt["det_type"]

                ### Calculate size of the metastasis ###
                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                bb_met_image = np.zeros(gt_mask.shape)
                bb_met_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                if det_type == 'det_tp':
                    ind = find_ind(gt_mask_labelled, bb_met_image)
                    keep_tp_ind.append(ind)
                    met = np.multiply(gt_mask_labelled == ind, 1)
                    maj_ax_le = obtain_size_met(met)
                    if maj_ax_le > 6:
                        TP_large = TP_large + 1
                    else:
                        TP_small = TP_small + 1

                elif det_type == 'det_fp':
                    datafile = os.path.join(pth_pred, patient_name)
                    pred_mask = np.squeeze(np.load(datafile))  # probability map
                    pred_mask = np.multiply(pred_mask > prob_map_thr, 1)
                    pred_mask = np.multiply(bb_met_image, pred_mask)  # mask of only one fp metastasis
                    maj_ax_le = obtain_size_met(pred_mask)
                    if maj_ax_le > 6:
                        FP_large = FP_large + 1
                    else:
                        FP_small = FP_small + 1

        # Calculate FN
        for i in range(1, np.max(gt_mask_labelled)+1):
            if i not in keep_tp_ind:
                met = gt_mask_labelled == i  #FN metastasis
                maj_ax_le = obtain_size_met(met)
                if maj_ax_le > 6:
                    FN_large = FN_large + 1
                else:
                    FN_small = FN_small + 1

        count_TP_large = count_TP_large + TP_large
        count_FP_large = count_FP_large + FP_large
        count_FN_large = count_FN_large + FN_large

        count_TP_small = count_TP_small + TP_small
        count_FP_small = count_FP_small + FP_small
        count_FN_small = count_FN_small + FN_small

        print("The total number of TP large in this patient is...", TP_large)
        print("The total number of FP large in this patient is...", FP_large)
        print("The total number of FN large in this patient is...", FN_large)

        print("The total number of TP small in this patient is...", TP_small)
        print("The total number of FP small in this patient is...", FP_small)
        print("The total number of FN small in this patient is...", FN_small)

    print("The total number of TP small is...",  count_TP_large)
    print("The total number of FP small is...", count_FP_large)
    print("The total number of FN small is...", count_FN_large)

    print("The total number of TP small is...",  count_TP_small)
    print("The total number of FP small is...", count_FP_small)
    print("The total number of FN small is...", count_FN_small)

    # Calculate precision and recall
    recall_large = count_TP_large/(count_TP_large + count_FN_large)
    recall_small = count_TP_small/(count_TP_small + count_FN_small)
    precision_large = count_TP_large/(count_TP_large + count_FP_large)
    precision_small = count_TP_small/(count_TP_small + count_FP_small)

    print("The recall large is...", recall_large)
    print("The recall small is...", recall_small)
    print("The precision large is...", precision_large)
    print("The precision small is...", precision_small)

    # Calculate F1 & accuracy
    F1_large = 2 * precision_large * recall_large / (precision_large + recall_large)
    print("The F1 score large is...", F1_large)
    F1_small = 2 * precision_small * recall_small / (precision_small + recall_small)
    print("The F1 score small is...", F1_small)

    # Calculate accuracy
    acc_large = count_TP_large / (count_TP_large + count_FP_large + count_FN_large)
    print("The accuracy large is...", acc_large)
    acc_small = count_TP_small / (count_TP_small + count_FP_small + count_FN_small)
    print("The accuracy is...", acc_small)

#######################################################################################################################
# FIND THRESHOLD FOR CONFIDENCE SCORE OF BOUNDING BOXES
#######################################################################################################################
if determine_thrf:
    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    thr_vals = np.arange(0.5, 1, 0.01)  # test thresholds in the range 0.5 - 1
    recall = np.zeros((1, len(thr_vals)))
    precision = np.zeros((1, len(thr_vals)))
    F1 = np.zeros((1, len(thr_vals)))

    for num, thr in enumerate(thr_vals):

        print("=====> Evaluating threshold...", thr)
        # Define files
        file_bb = file_bb[file_bb["score"] > thr]

        count_TP = 0
        count_FP = 0
        count_FN = 0

        for ipatient in range(1, n_patients + 1):

            # Initialize values
            TP = 0
            FN = 0
            FP = 0

            print("Evaluating patient... ", ipatient)
            patient_name = str(ipatient) + '.npy'
            df_bb = file_bb[file_bb["patientID"] == ipatient]
            df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

            # Calculate TP and FP
            if df_bb.empty:
                print("No detected metastases for this patient")
            else:
                for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                    temp_image = np.zeros((384, 384, 272))
                    met_bb = df_bb.iloc[met_num, :]
                    a = np.where(np.round(df_det_type["pred_score"], decimals=6) ==np.round(met_bb["score"], decimals=6))
                    if np.array(a).shape[1] == 0:
                        a = np.where(np.round(df_det_type["pred_score"], decimals=4) == np.round(met_bb["score"], decimals0=4))
                    elif np.array(a).shape[1] > 1:
                        print("It requires more decimal figures for unique assigment")
                        a = np.where(np.round(df_det_type["pred_score"], decimals=10) == np.round(met_bb["score"], decimals=10))
                        a = [0]
                    a = a[0]

                    met_dt = df_det_type.iloc[int(a), :]
                    met_coords = met_bb["coords"]
                    met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                    met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                    temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                    det_type = met_dt["det_type"]

                    if det_type == 'det_tp':
                        TP = TP + 1

                    elif det_type == 'det_fp':
                        if met_bb["score"] > 0.9:  # may be change this to 0.8
                            print("Met num is...", met_num)
                            # Check that indeed is fp: if with the predictionID says TP and IoU>0.1, it will be TP
                            met_dt = df_det_type.iloc[met_bb["predictionID"], :]  # check that predictionID says TP
                            det_type_predID = met_dt["det_type"]
                            if det_type_predID == 'det_tp':
                                print("performing checks")
                                # Load GT mask
                                datafile = os.path.join(pth_gt, patient_name)
                                gt_data = np.load(datafile)
                                gt_mask = gt_data[1, :, :, :]
                                gt_mask_labelled = measure.label(gt_mask)
                                iou_temp_max = 0
                                for x in range(1, np.max(gt_mask_labelled)+1):
                                    gt_met = gt_mask_labelled == x
                                    gt_bb = np.zeros(gt_mask.shape)
                                    gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                                    iou_temp = compute_iou_patient(gt_bb, temp_image)
                                    print(iou_temp)
                                    if iou_temp > iou_temp_max:
                                        iou_temp_max = iou_temp
                                if iou_temp_max > 0.1:
                                    print("It is true positive")
                                    TP = TP + 1
                                else:
                                    FP = FP + 1
                        else:
                            FP = FP + 1

            # Calculate FN
            df_det_type_met = df_det_type[df_det_type["class_label"] == 1]  # Find predictionsIDs that are metastases
            predictionsIDs = np.array(df_det_type_met.T.columns) - np.array(df_det_type.T.columns)[0]
            gt_num_met = len(predictionsIDs)
            FN = gt_num_met - TP

            count_TP = count_TP + TP
            count_FP = count_FP + FP
            count_FN = count_FN + FN

            print("The total number of TP in this patient is...", TP)
            print("The total number of FP in this patient is...", FP)
            print("The total number of FN in this patient is...", FN)

        print("The total number of TP is...",  count_TP)
        print("The total number of FP is...", count_FP)
        print("The total number of FN is...", count_FN)

        # Calculate precision and recall
        recall[0, num] = count_TP/(count_TP + count_FN)
        precision[0, num] = count_TP/(count_TP + count_FP)

        print("The recall is...", recall[0, num])
        print("The precision is...", precision[0, num])

        # # Calculate F1 & accuracy
        F1[0, num] = 2 * precision[0, num] * recall[0, num] / (precision[0, num] + recall[0, num])
        print("The F1 score is...", F1)

        acc = count_TP / (count_TP + count_FP + count_FN)
        print("The accuracy is...", acc)

    print("The largest F1 score is...", np.max(F1))

    # CHANGE THE NAMES ACCORDING TO THE NETWORK YOU ARE EVALUATING:
    datafile = 'recall_detection_Unet.npy'
    np.save(datafile, recall)
    datafile = 'precision_detection_Unet.npy'
    np.save(datafile, precision)
    datafile = 'F1_detection_Unet.npy'
    np.save(datafile, F1)

######################################################################################################################
# Calculate graph volume
######################################################################################################################

if volume_graph:
    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    count_TP = 0
    count_FP = 0
    count_FN = 0

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]

    # Create variables for saving the GT and predicted volume
    volume_data = []
    count_vol = 0  # keep count of TP metastases to include in the volume measurement
    factor = 0.001  # Factor to pass from mm^3 to ml

    for ipatient in range(1, n_patients + 1):

        image_bb_tp = np.zeros((384, 384, 272))

        # Initialize values
        TP = 0
        FN = 0
        FP = 0

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) == np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]

                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                image_bb_tp[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]
                if det_type == 'det_tp':
                    TP = TP + 1
                    count_vol = count_vol + 1

                    # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                    datafile = os.path.join(pth_gt, patient_name)
                    gt_data = np.load(datafile)
                    gt_mask = gt_data[1, :, :, :]
                    gt_mask_labelled = measure.label(gt_mask)
                    iou_temp_max = 0
                    ind = 0
                    for x in range(1, np.max(gt_mask_labelled) + 1):
                        gt_met = gt_mask_labelled == x
                        gt_bb = np.zeros(gt_mask.shape)
                        gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]), np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]), np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                        iou_temp = compute_iou_patient(gt_bb, temp_image)
                        print(iou_temp)
                        if iou_temp > iou_temp_max:
                            iou_temp_max = iou_temp
                            ind = x
                    # Calculate volume gt
                    gt_met = gt_mask_labelled == ind
                    num_voxels = np.multiply(np.sum(gt_met), 1)  # it is in mm^3
                    gt_vol = num_voxels * factor

                    # Calculate predicted volume
                    # Load segmented mask
                    datafile = os.path.join(pth_pred, patient_name)
                    pred_mask = np.squeeze(np.load(datafile))  # probability map
                    pred_mask = np.multiply(pred_mask > 0.4, 1)
                    # Multiply the predicted mask with the temp_image that has the bunding box of the metastasis of interest
                    pred_mask = np.multiply(pred_mask, temp_image)
                    # Choose the greatest object inside (in case there are several)
                    pred_mask_labelled = measure.label(pred_mask)
                    num_objs = np.max(pred_mask_labelled)
                    if num_objs > 1:
                        print("The number of objects is greater than 1")
                    num_voxels = np.multiply(np.sum(pred_mask), 1)  # it is in mm^3
                    pred_vol = num_voxels * factor
                    var = [gt_vol, pred_vol]
                    volume_data.append(var)
                    print(volume_data)
                    print("The GT volume is...", gt_vol)
                    print("The predicted volume is...", pred_vol)

                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.5, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :]  # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")
                            TP = TP + 1

                            count_vol = count_vol + 1

                            # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                            datafile = os.path.join(pth_gt, patient_name)
                            gt_data = np.load(datafile)
                            gt_mask = gt_data[1, :, :, :]
                            gt_mask_labelled = measure.label(gt_mask)
                            iou_temp_max = 0
                            ind = 0
                            for x in range(1, np.max(gt_mask_labelled) + 1):
                                gt_met = gt_mask_labelled == x
                                gt_bb = np.zeros(gt_mask.shape)
                                gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),
                                np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),
                                np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                                iou_temp = compute_iou_patient(gt_bb, temp_image)
                                print(iou_temp)
                                if iou_temp > iou_temp_max:
                                    iou_temp_max = iou_temp
                                    ind = x
                            # Calculate volume gt
                            gt_met = gt_mask_labelled == ind
                            num_voxels = np.multiply(np.sum(gt_met), 1)  # it is in mm^3
                            gt_vol = num_voxels * factor

                            # Calculate predicted volume
                            # Load segmented mask
                            datafile = os.path.join(pth_pred, patient_name)
                            pred_mask = np.squeeze(np.load(datafile))  # probability map
                            pred_mask = np.multiply(pred_mask > 0.4, 1)
                            # Multiply the predicted mask with the temp_image that has the bunding box of the metastasis of interest
                            pred_mask = np.multiply(pred_mask, temp_image)
                            # Choose the greatest object inside (in case there are several)
                            pred_mask_labelled = measure.label(pred_mask)
                            num_objs = np.max(pred_mask_labelled)
                            if num_objs > 1:
                                print("The number of objects is greater than 1")
                            num_voxels = np.multiply(np.sum(pred_mask), 1)  # it is in mm^3
                            pred_vol = num_voxels * factor
                            #volume_data[count_vol, 1] = pred_vol
                            var = [gt_vol, pred_vol]
                            volume_data.append(var)
                            print("The GT volume is...", gt_vol)
                            print("The predicted volume is...", pred_vol)
                        else:
                            FP = FP + 1
                    else:
                        FP = FP + 1

    print(volume_data)
    output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/' + 'volume_data'
    sio.savemat('volume_data.mat', {'volume_data': volume_data})
    np.save(output_datafile + '.npy', volume_data)

###################################################################################################################
# Obtain variables for histogram with all GT metastases and classify them in TP and FN
###################################################################################################################

if missed_and_found:

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    count_TP = 0
    count_FP = 0
    count_FN = 0

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]

    # Create variables for saving the GT and predicted volume
    volume_data_GT = []  # first column will be the GT volume and second column will be the label ( 1: TP and 2: FN)
    count_vol = 0  # keep count of TP metastases to include in the volume measurement
    factor = 0.001  # Factor to pass from mm^3 to ml

    for ipatient in range(1, n_patients + 1):
        indices_GT = []
        image_bb_tp = np.zeros((384, 384, 272))

        # Initialize values
        TP = 0
        FN = 0
        FP = 0

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) == np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]  # -1 added new

                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                image_bb_tp[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]
                if det_type == 'det_tp':
                    TP = TP + 1
                    count_vol = count_vol + 1

                    # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                    datafile = os.path.join(pth_gt, patient_name)
                    gt_data = np.load(datafile)
                    gt_mask = gt_data[1, :, :, :]
                    gt_mask_labelled = measure.label(gt_mask)
                    iou_temp_max = 0
                    ind = 0
                    for x in range(1, np.max(gt_mask_labelled) + 1):
                        gt_met = gt_mask_labelled == x
                        gt_bb = np.zeros(gt_mask.shape)
                        gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]), np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]), np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                        iou_temp = compute_iou_patient(gt_bb, temp_image)
                        print(iou_temp)
                        if iou_temp > iou_temp_max:
                            iou_temp_max = iou_temp
                            ind = x
                    # Calculate volume gt
                    indices_GT.append(ind)
                    gt_met = gt_mask_labelled == ind
                    num_voxels = np.multiply(np.sum(gt_met), 1)  # it is in mm^3
                    gt_vol = num_voxels * factor
                    var = [gt_vol, 1]
                    volume_data_GT.append(var)
                    print(volume_data_GT)
                    print("The GT volume is...", gt_vol)

                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.5, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :] # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")
                            TP = TP + 1

                            count_vol = count_vol + 1

                            # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                            datafile = os.path.join(pth_gt, patient_name)
                            gt_data = np.load(datafile)
                            gt_mask = gt_data[1, :, :, :]
                            gt_mask_labelled = measure.label(gt_mask)
                            iou_temp_max = 0
                            ind = 0
                            for x in range(1, np.max(gt_mask_labelled) + 1):
                                gt_met = gt_mask_labelled == x
                                gt_bb = np.zeros(gt_mask.shape)
                                gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),
                                np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),
                                np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                                iou_temp = compute_iou_patient(gt_bb, temp_image)
                                print(iou_temp)
                                if iou_temp > iou_temp_max:
                                    iou_temp_max = iou_temp
                                    ind = x
                            # Calculate volume gt
                            indices_GT.append(ind)
                            gt_met = gt_mask_labelled == ind
                            num_voxels = np.multiply(np.sum(gt_met), 1)  # it is in mm^3
                            gt_vol = num_voxels * factor
                            var = [gt_vol, 1]
                            volume_data_GT.append(var)
                            print("The GT volume is...", gt_vol)
                        else:
                            FP = FP + 1
                    else:
                        FP = FP + 1
        #Calculate FN
        # Load GT mask: find the corresponding metastasis (pair GT and predicted)
        datafile = os.path.join(pth_gt, patient_name)
        gt_data = np.load(datafile)
        gt_mask = gt_data[1, :, :, :]
        gt_mask_labelled = measure.label(gt_mask)
        indices_labelled = np.unique(gt_mask_labelled)
        indices_labelled = indices_labelled[1:]
        print("Indices labelled are...", indices_labelled)
        print("Indices GT are...", indices_GT)
        fn_ind = np.isin(indices_labelled, indices_GT)
        fn_ind = list(map(operator.not_, fn_ind))  # FN entries will have the value True instead of False
        print(fn_ind)
        fn_ind = np.multiply(indices_labelled, np.multiply(fn_ind, 1))
        print(indices_labelled)
        print(fn_ind)
        for kk in range(len(indices_labelled)):
            val = fn_ind[kk]
            if val > 0:  # FN
                gt_met = gt_mask_labelled == kk + 1
                num_voxels = np.multiply(np.sum(gt_met), 1)  # it is in mm^3
                gt_vol = num_voxels * factor
                var = [gt_vol, 2]
                volume_data_GT.append(var)
                print("The GT volume of FN is...", gt_vol)

    print(volume_data_GT)
    output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/' + 'volume_data_GT'
    sio.savemat('volume_data_GT.mat', {'volume_data_GT': volume_data_GT})
    np.save(output_datafile + '.npy', volume_data_GT)


################################################################################################################
# Create box-whisker plots demonstrating the final per-metastasis level segmentation (only TP)
################################################################################################################

if whisker_plot:

    prob_map_thr = 0.4  # Threshold for the probability map output determined through a validation set
    # 0.4 (Original Mask R-CNN implementation)
    # 0.3 (Mask R-CNN with ResNet101 backbone)
    # 0.5 (Mask R-CNN config 3 (see report))
    # 0.1 (Retina U-Net)
    # 0.4 (Detection U-Net)

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    factor = 0.001

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]

    # Create variables for saving the GT and predicted volume
    dice_stratified = []  # First column: GT met. volume. Second col.: pred. met volume. The third the dice coefficient

    for ipatient in range(1, n_patients + 1):
        indices_GT = []
        # Initialize values

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) == np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]  # -1 added new

                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]
                if det_type == 'det_tp':

                    # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                    datafile = os.path.join(pth_gt, patient_name)
                    gt_data = np.load(datafile)
                    gt_mask = gt_data[1, :, :, :]
                    gt_mask_labelled = measure.label(gt_mask)
                    iou_temp_max = 0
                    ind = 0
                    for x in range(1, np.max(gt_mask_labelled) + 1):
                        gt_met = gt_mask_labelled == x
                        gt_bb = np.zeros(gt_mask.shape)
                        gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]), np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]), np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                        iou_temp = compute_iou_patient(gt_bb, temp_image)
                        print(iou_temp)
                        if iou_temp > iou_temp_max:
                            iou_temp_max = iou_temp
                            ind = x
                    # Calculate volume gt
                    indices_GT.append(ind)
                    gt_met = gt_mask_labelled == ind
                    gt_vol = np.sum(np.multiply(gt_met, 1)) * 0.001  # Volume in mL

                    #Calculate predicted volume
                    # Load segmented mask
                    datafile = os.path.join(pth_pred, patient_name)
                    pred_mask = np.squeeze(np.load(datafile))  # probability map
                    pred_mask = np.multiply(pred_mask > prob_map_thr, 1)
                    # Multiply the predicted mask with the temp_image that has the bunding box of the metastasis of interest
                    pred_mask = np.multiply(pred_mask, temp_image)
                    # Choose the greatest object inside (in case there are several)
                    pred_mask_labelled = measure.label(pred_mask)
                    num_objs = np.max(pred_mask_labelled)
                    if num_objs > 1:
                        print("The number of objects is greater than 1")
                    num_voxels = np.multiply(np.sum(pred_mask), 1)  # it is in mm^3
                    pred_vol = num_voxels * factor
                    dice_val = compute_dice_patient(pred_mask, np.multiply(gt_met, 1))
                    var = [gt_vol, pred_vol, dice_val]
                    print(var)
                    dice_stratified.append(var)

                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.5, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :] # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")

                            # Load GT mask: find the corresponding metastasis (pair GT and predicted)
                            datafile = os.path.join(pth_gt, patient_name)
                            gt_data = np.load(datafile)
                            gt_mask = gt_data[1, :, :, :]
                            gt_mask_labelled = measure.label(gt_mask)
                            iou_temp_max = 0
                            ind = 0
                            for x in range(1, np.max(gt_mask_labelled) + 1):
                                gt_met = gt_mask_labelled == x
                                gt_bb = np.zeros(gt_mask.shape)
                                gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),
                                np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),
                                np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
                                iou_temp = compute_iou_patient(gt_bb, temp_image)
                                print(iou_temp)
                                if iou_temp > iou_temp_max:
                                    iou_temp_max = iou_temp
                                    ind = x
                            # Calculate volume gt
                            indices_GT.append(ind)
                            gt_met = gt_mask_labelled == ind
                            gt_vol = np.sum(np.multiply(gt_met, 1)) * 0.001  # Volume in mL

                            # Calculate predicted volume
                            # Load segmented mask
                            datafile = os.path.join(pth_pred, patient_name)
                            pred_mask = np.squeeze(np.load(datafile))  # probability map
                            pred_mask = np.multiply(pred_mask > 0.4, 1)
                            # Multiply the predicted mask with the temp_image that has the bunding box of the metastasis of interest
                            pred_mask = np.multiply(pred_mask, temp_image)
                            # Choose the greatest object inside (in case there are several)
                            pred_mask_labelled = measure.label(pred_mask)
                            num_objs = np.max(pred_mask_labelled)
                            if num_objs > 1:
                                print("The number of objects is greater than 1")
                            num_voxels = np.multiply(np.sum(pred_mask), 1)  # it is in mm^3
                            pred_vol = num_voxels * factor
                            dice_val = compute_dice_patient(pred_mask, np.multiply(gt_met, 1))
                            var = [gt_vol, pred_vol, dice_val]
                            print(var)
                            dice_stratified.append(var)
    output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/' + 'dice_stratified'
    sio.savemat('dice_stratified.mat', {'dice_stratified': dice_stratified})
    np.save(output_datafile + '.npy', dice_stratified)

#######################################################################################################################
# Calculate the average recall and precision per patient
#######################################################################################################################
if per_patient:

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder

    precision_all = np.zeros((1, 54))
    recall_all = np.zeros((1, 54))

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]
    for ipatient in range(1, n_patients + 1):

        # Initialize values (TP, FN and FP per patient)
        TP = 0
        FN = 0
        FP = 0

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) == np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]  # -1 added new

                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]
                if det_type == 'det_tp':
                    TP = TP + 1

                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.5, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :] # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")
                            TP = TP + 1
                        else:
                            FP = FP + 1
                    else:
                        FP = FP + 1

        # Calculate FN
        df_det_type_met = df_det_type[df_det_type["class_label"] == 1]  # Find predictionsIDs that are metastases
        predictionsIDs = np.array(df_det_type_met.T.columns) - np.array(df_det_type.T.columns)[0]
        gt_num_met = len(predictionsIDs)

        FN = gt_num_met - TP

        # Calculate precision and recall
        recall_all[0, int(ipatient) - 1] = TP/(TP + FN)
        if TP + FP > 0:
            precision_all[0, int(ipatient) - 1] = TP/(TP + FP)
        else:
            precision_all[0, int(ipatient) - 1] = np.NAN
        print("The total number of TP in this patient is...", TP)
        print("The total number of FP in this patient is...", FP)
        print("The total number of FN in this patient is...", FN)
        print("The recall is...", recall_all[0, int(ipatient) - 1])
        print("The precision is...", precision_all[0, int(ipatient) - 1])

    # Print mean
    print("The recall  is...", np.nanmean(np.array(recall_all)))
    print("The precision  is...", np.nanmean(np.array(precision_all)))

    # Print standard deviation
    print("The recall standard deviation is...", np.nanstd(np.array(recall_all)))
    print("The precision standard deviation is...", np.nanstd(np.array(precision_all)))


#######################################################################################################################
# Calculate FP per patient
#######################################################################################################################

if fp_per_patient:

    # Calculate the number of TP, FP and FN
    n_patients = 54  # 54 patients in test folder
    count_TP = 0
    count_FP = 0
    count_FN = 0
    store_FP = np.zeros((1, 54))

    # Define files
    file_bb = file_bb[file_bb["score"] > thrf]
    for ipatient in range(1, n_patients + 1):

        image_bb_tp = np.zeros((384, 384, 272))

        # Initialize values
        TP = 0
        FN = 0
        FP = 0

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        df_bb = file_bb[file_bb["patientID"] == ipatient]
        df_det_type = file_det_type[file_det_type["pid"] == str(ipatient)]

        # Calculate TP and FP
        if df_bb.empty:
            print("No detected metastases for this patient")
        else:
            for met_num in range(len(df_bb)):  # Loop through all the metastases for a given patient
                temp_image = np.zeros((384, 384, 272))
                met_bb = df_bb.iloc[met_num, :]
                a = np.where(np.round(df_det_type["pred_score"], decimals=5) ==np.round(met_bb["score"], decimals=5))
                a = a[0]
                met_dt = df_det_type.iloc[int(a), :]  # -1 added new
                met_coords = met_bb["coords"]
                met_coords = met_coords[1:len(met_coords) - 1]  # remove brackets
                met_coords = np.floor(np.fromstring(met_coords, dtype=np.float, sep=',')).astype('int')
                temp_image[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1
                image_bb_tp[met_coords[0]:met_coords[2], met_coords[1]:met_coords[3], met_coords[4]:met_coords[5]] = 1

                det_type = met_dt["det_type"]
                if det_type == 'det_tp':
                    TP = TP + 1

                elif det_type == 'det_fp':
                    print("Met num is...", met_num)
                    # Check that indeed is fp: if with the predictionID says TP and IoU>0.5, it will be TP
                    met_dt = df_det_type.iloc[met_bb["predictionID"], :] # check that predictionID says TP
                    det_type_predID = met_dt["det_type"]
                    if det_type_predID == 'det_tp':
                        print("performing checks")
                        # Load GT mask
                        datafile = os.path.join(pth_gt, patient_name)
                        gt_data = np.load(datafile)
                        gt_mask = gt_data[1, :, :, :]
                        gt_mask_labelled = measure.label(gt_mask)
                        iou_temp_max = 0
                        for x in range(1,np.max(gt_mask_labelled)+1):
                            gt_met = gt_mask_labelled == x
                            gt_bb = np.zeros(gt_mask.shape)
                            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])]=1
                            iou_temp = compute_iou_patient(gt_bb, temp_image)
                            print(iou_temp)
                            if iou_temp > iou_temp_max:
                                iou_temp_max = iou_temp
                        if iou_temp_max > 0.1:
                            print("It is true positive")
                            TP = TP + 1
                        else:
                            FP = FP + 1
                    else:
                        FP = FP + 1

        store_FP[0, ipatient-1] = FP

    print("The average number of FP per patient is...", np.mean(store_FP))
    print("The standard deviation of FP per patient is...", np.std(store_FP))
    print("total number of FP...", np.sum(store_FP))





