
######################################################################################################################
# Cristina Almagro Perez, 2022. ETH University.
######################################################################################################################

# Code to calculate the DSC and IoU of the GT and output predicted segmentation masks

# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy.io as sio
from skimage import measure

######################################################################################################################
# Auxiliary functions
######################################################################################################################

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

#######################################################################################################################


##### SPECIFY THE FOLLOWING: ######
n_patients = 54
prob_map_thr = 0.4  # Threshold for the probability map output determined through a validation set
    # 0.4 (Original Mask R-CNN implementation)
    # 0.3 (Mask R-CNN with ResNet101 backbone)
    # 0.5 (Mask R-CNN config 3 (see report))
    # 0.1 (Retina U-Net)
    # 0.4 (Detection U-Net)
average_metrics_patient = True  # Calculate the average DSC and/or IoU per patient
metrics_TP = True  # Obtain the average DSC and/or IoU for TP metastases averaging over all TP met. and not patient
average_metrics_TP = True  # Calculate the average DSC and/or IoU per patient considering only TP metastases
stratified_DICE = True  # Obtain DICE value for metastases smaller than 6mm and metastases larger than 6mm
find_prob_map_thr = True  # Find threshold of the output probability map for metastases
find_prob_map_thr_tp = True  # Find threshold of the output probability map for metastases (only TP metastases)

# Finally, select the metrics for each analysis of the above
compute_dice = True
compute_iou = False


# Define path for the location of the GT masks and predicted masks
pth_gt = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test0'
datafile = os.path.join(pth_gt, 'info_df.pickle')
info_df = pd.read_pickle(datafile)
dice_all = 0
iou_all = 0

# Paths for original 3D Mask R-CNN implementation:
pth_pred = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/test_pred_seg/all_patients'
pth_bb_masks = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_hold_out_min_det_0.72/enlarged_5'
pth_bb_masks_tp = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_hold_out_min_det_0.9/only_tp_bb'

# Paths for ResNet101
# pth_pred = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_1_ResNet101/test_pred_seg/'
# pth_bb_masks = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_1_ResNet101/results_hold_out_min_det_0.69/enlarged_5'
# pth_bb_masks_tp= r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/ResNet101/results_hold_out_min_det_0.69/only_tp_bb/'

# Paths param 300
# pth_pred = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_0_param_300/test_pred_seg/'
# pth_bb_masks = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/results_fold_0_param_300/results_hold_out_min_det_0.79/enlarged_5'
# pth_bb_masks_tp = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/param300/results_hold_out_min_det_0.79/only_tp_bb/'

# Paths retina unet
# pth_pred = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/retina_unet/test_pred_seg/'
# pth_bb_masks = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/retina_unet/results_hold_out_min_det_0.79/enlarged_5/'
# pth_bb_masks_tp = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/retina_unet/results_hold_out_min_det_0.79/only_tp_bb/'

# Paths detection unet
pth_pred = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/detection_unet/test_pred_seg/'
pth_bb_masks = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/detection_unet/results_hold_out_min_det_0.70/enlarged_5/'


dice_vals = np.zeros((1, 54))  # store values for later calculation of the standard deviation
iou_vals = np.zeros((1, 54))  # store values for later calculation of the standard deviation
#

######################################################################################################################
# Calculate the average DSC and/or IoU per patient
######################################################################################################################

if average_metrics_patient:

    for ipatient in info_df["pid"]:

        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'

        # Load GT mask
        datafile = os.path.join(pth_gt, patient_name)
        gt_mask = np.load(datafile)
        gt_mask = gt_mask[1, :, :, :]
        # Load predicted mask
        datafile = os.path.join(pth_pred, patient_name)
        pred_mask = np.squeeze(np.load(datafile))  # probability map
        pred_mask = np.multiply(pred_mask > prob_map_thr, 1)
        # Load bounding box mask
        datafile = os.path.join(pth_bb_masks, patient_name)
        bb_mask = np.load(datafile)
        pred_mask = np.multiply(bb_mask, pred_mask)

        if compute_dice:
            dice_value = compute_dice_patient(pred_mask, gt_mask)
            dice_all = dice_all + dice_value
            dice_vals[0, int(ipatient)-1] = dice_value
            print(dice_value)

        # Compute IoU
        if compute_iou:
            iou_value = compute_iou_patient(pred_mask, gt_mask)
            iou_all = iou_all + iou_value
            iou_vals[0, int(ipatient) - 1] = iou_value
            print(iou_value)

    # Display values
    if compute_dice:
        dice_all = dice_all/n_patients
        print("The DICE value is...", dice_all)
    if compute_iou:
        iou_all = iou_all/n_patients
        print("The IoU value is...", iou_all)

    #Save dice and iou values
    # sio.savemat('dice_vals.mat', {'dice_vals': dice_vals})
    # sio.savemat('iou_vals.mat', {'iou_vals': iou_vals})

    # Print standard deviation
    print("The DSC standard deviation is...", np.std(dice_vals))
    print("The IoU standard deviation is...", np.std(iou_vals))

######################################################################################################################
# Evaluate DSC and/or IoU for metastases that have been correctly detected (TP)
######################################################################################################################

if metrics_TP:
    count_TP = 0
    dice_vals_tp = []  # store values for later calculation of the standard deviation
    iou_vals_tp = []  # store values for later calculation of the standard deviation

    for ipatient in info_df["pid"]:
        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'

        # Load GT mask
        datafile = os.path.join(pth_gt, patient_name)
        gt_mask = np.load(datafile)
        gt_mask = gt_mask[1, :, :, :]
        gt_mask_labelled = measure.label(gt_mask)

        # Load predicted mask
        datafile = os.path.join(pth_pred, patient_name)
        pred_mask = np.squeeze(np.load(datafile))  # probability map
        pred_mask = np.multiply(pred_mask > 0.5, 1)  # 0.1 for Retina Unet, # 0.3 for ResNet101, 0.5 for param300

        # Load bounding box mask (only TP)
        datafile = os.path.join(pth_bb_masks_tp, patient_name)
        bb_mask = np.load(datafile)
        pred_mask = np.multiply(bb_mask, pred_mask)  # It will contain only the metastases inside a TP
        pred_mask_labelled = measure.label(pred_mask)

        for i in range(np.max(pred_mask_labelled)):
            val = i + 1
            met_pred_mask = np.multiply(pred_mask_labelled == val, 1)

            index = 0
            max_dice = 0
            max_iou = 0
            for j in range(np.max(gt_mask_labelled)):
                met_gt_mask = np.multiply(gt_mask_labelled == j + 1, 1)

                if compute_dice:
                    dice_value = compute_dice_patient(met_pred_mask, met_gt_mask)
                    if dice_value > max_dice:
                        max_dice = dice_value

                if compute_iou:
                    iou_value = compute_iou_patient(met_pred_mask, met_gt_mask)
                    if iou_value > max_iou:
                        max_iou = iou_value

            if (max_dice > 0) | (max_iou > 0):
                count_TP = count_TP + 1
                dice_all = dice_all + max_dice
                iou_all = iou_all + max_iou
                dice_vals_tp.append(max_dice)
                iou_vals_tp.append(max_iou)

    print("count_TP is ...", count_TP)
    # Display values
    if compute_dice:
        dice_all = dice_all/count_TP
        print("The DICE value is...", dice_all)
    if compute_iou:
        iou_all = iou_all/count_TP
        print("The IoU value is...", iou_all)

    # Save dice and iou values
    sio.savemat('dice_vals_tp.mat', {'dice_vals_tp': dice_vals_tp})
    sio.savemat('iou_vals_tp.mat', {'iou_vals_tp': iou_vals_tp})

    # Print standard deviation
    print("The DSC standard deviation is...", np.std(np.array(dice_vals_tp)))
    print("The IoU standard deviation is...", np.std(np.array(iou_vals_tp)))

#######################################################################################################################
# Evaluate DICE coefficient for metastases that have been correctly detected (TP)
#######################################################################################################################
if average_metrics_TP:
    count_TP = 0
    dice_vals_tp = np.zeros((1, 54))  # store values for later calculation of the standard deviation
    iou_vals_tp = np.zeros((1, 54))  # store values for later calculation of the standard deviation

    for ipatient in info_df["pid"]:
        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'
        dice_all = 0
        iou_all = 0

        # Load GT mask
        datafile = os.path.join(pth_gt, patient_name)
        gt_mask = np.load(datafile)
        gt_mask = gt_mask[1, :, :, :]
        gt_mask_labelled = measure.label(gt_mask)

        # Load predicted mask
        datafile = os.path.join(pth_pred, patient_name)
        pred_mask = np.squeeze(np.load(datafile))  # probability map
        pred_mask = np.multiply(pred_mask > 0.4, 1)

        # Load bounding box mask (only TP)
        datafile = os.path.join(pth_bb_masks_tp, patient_name)
        bb_mask = np.load(datafile)
        pred_mask = np.multiply(bb_mask, pred_mask)  # It will contain only the metastases inside a TP
        pred_mask_labelled = measure.label(pred_mask)

        for i in range(np.max(pred_mask_labelled)):
            val = i + 1
            met_pred_mask = np.multiply(pred_mask_labelled == val, 1)

            index = 0
            max_dice = 0
            max_iou = 0
            for j in range(np.max(gt_mask_labelled)):
                met_gt_mask = np.multiply(gt_mask_labelled == j + 1, 1)

                if compute_dice:
                    dice_value = compute_dice_patient(met_pred_mask, met_gt_mask)
                    if dice_value > max_dice:
                        max_dice = dice_value

                if compute_iou:
                    iou_value = compute_iou_patient(met_pred_mask, met_gt_mask)
                    if iou_value > max_iou:
                        max_iou = iou_value

            if (max_dice > 0) | (max_iou > 0):
                count_TP = count_TP + 1
                dice_all = dice_all + max_dice
                iou_all = iou_all + max_iou
        if np.max(pred_mask_labelled) > 0:  # at least one tp
            dice_vals_tp[0, int(ipatient) - 1] = dice_all/np.max(pred_mask_labelled)
            iou_vals_tp[0, int(ipatient) - 1] = iou_all/np.max(pred_mask_labelled)
        else:
            dice_vals_tp[0, int(ipatient) - 1] = np.NAN
            iou_vals_tp[0, int(ipatient) - 1] = np.NAN

    # Save dice and iou values
    # sio.savemat('dice_vals_tp.mat', {'dice_vals_tp': dice_vals_tp})
    # sio.savemat('iou_vals_tp.mat', {'iou_vals_tp': iou_vals_tp})

    # Print mean
    print("The DSC  is...", np.nanmean(np.array(dice_vals_tp)))
    print("The IoU  is...", np.nanmean(np.array(iou_vals_tp)))

    # Print standard deviation
    print("The DSC standard deviation is...", np.nanstd(np.array(dice_vals_tp)))
    print("The IoU standard deviation is...", np.nanstd(np.array(iou_vals_tp)))

#############################################################################################################################################
# Find the optimal probability map threshold: considering all metastases
#############################################################################################################################################
if find_prob_map_thr:
    print("FIND OPTIMAL VALUE OF DSC FOR ALL METASTASES")
    thrs = np.linspace(0, 1, num=11)
    ind = 0
    dice_max_thr = 0

    for kk in thrs:
        print("Evaluating threshold... ", kk)
        dice_all = 0
        count_TP = 0
        for ipatient in info_df["pid"]:
            print("Evaluating patient... ", ipatient)
            patient_name = str(ipatient) + '.npy'

            # Load GT mask
            datafile = os.path.join(pth_gt, patient_name)
            gt_mask = np.load(datafile)
            gt_mask = gt_mask[1, :, :, :]

            # Load predicted mask
            datafile = os.path.join(pth_pred, patient_name)
            pred_mask = np.squeeze(np.load(datafile))  # probability map
            pred_mask = np.multiply(pred_mask > kk, 1)
            # Load bounding box mask
            datafile = os.path.join(pth_bb_masks, patient_name)
            bb_mask = np.load(datafile)
            pred_mask = np.multiply(bb_mask, pred_mask)

            # Compute DICE
            dice_value = compute_dice_patient(pred_mask, gt_mask)
            dice_all = dice_all + dice_value
            print(dice_value)

        # Display values
        dice_all = dice_all/n_patients
        print("The DICE value is...", dice_all)
        if dice_all > dice_max_thr:
            dice_max_thr = dice_all
            ind = kk

    print("The optimal value (considering all metastases) for the threshold of the probability map is...", ind)
    print("For that value the DSC is ...", dice_max_thr)


#############################################################################################################################################
# Find the optimal probability map threshold: consider only TP
#############################################################################################################################################
if find_prob_map_thr_tp:

    thrs = np.linspace(0, 1, num=11)
    ind = 0
    dice_max_thr = 0

    for kk in thrs:
        print("Evaluating threshold... ", kk)
        dice_all = 0
        count_TP = 0
        for ipatient in info_df["pid"]:
            print("Evaluating patient... ", ipatient)
            patient_name = str(ipatient) + '.npy'

            # Load GT mask
            datafile = os.path.join(pth_gt, patient_name)
            gt_mask = np.load(datafile)
            gt_mask = gt_mask[1, :, :, :]
            gt_mask_labelled = measure.label(gt_mask)

            # Load predicted mask
            datafile = os.path.join(pth_pred, patient_name)
            pred_mask = np.squeeze(np.load(datafile))  # probability map
            pred_mask = np.multiply(pred_mask > kk, 1)


            # Load bounding box mask (only TP)
            datafile = os.path.join(pth_bb_masks_tp, patient_name)
            bb_mask = np.load(datafile)
            pred_mask = np.multiply(bb_mask, pred_mask)  # It will contain only the metastases inside a TP
            pred_mask_labelled = measure.label(pred_mask)

            for i in range(np.max(pred_mask_labelled)):
                val = i + 1
                met_pred_mask = np.multiply(pred_mask_labelled == val, 1)

                index = 0
                max_dice = 0

                for j in range(np.max(gt_mask_labelled)):
                    met_gt_mask = np.multiply(gt_mask_labelled == j + 1, 1)
                    dice_value = compute_dice_patient(met_pred_mask, met_gt_mask)
                    if dice_value > max_dice:
                        max_dice = dice_value

                if max_dice > 0:
                    count_TP = count_TP + 1
                    dice_all = dice_all + max_dice
                    print(max_dice)


        print("count_TP is ...", count_TP)
        # Display values
        dice_all = dice_all/count_TP
        print("The DICE value is...", dice_all)
        if dice_all > dice_max_thr:
            dice_max_thr = dice_all
            ind = kk

    print("The optimal value for the threshold of the probability map is...", ind)
    print("For that value the DSC is ...", ind)

##########################################################################################################
# Dice coefficient for metastases greater and smaller than 6 mm
##########################################################################################################
if stratified_DICE:
    # CHOOSE ONE OF THE FOLLOWING:
    option1 = False  # measuring the largest diameter in 3D
    option2 = False  # project in the craniocaudal direction and measure the largest diameter
    option3 = True  # project in the craniocaudal direction and measure the largest cross-sectional dimension

    count_large = 0
    count_small = 0
    dice_all_large = 0
    dice_all_small = 0
    iou_all_large = 0
    iou_all_small = 0

    for ipatient in info_df["pid"]:
        print("Evaluating patient... ", ipatient)
        patient_name = str(ipatient) + '.npy'

        # Load GT mask
        datafile = os.path.join(pth_gt, patient_name)
        gt_mask = np.load(datafile)
        gt_mask = gt_mask[1, :, :, :]
        gt_mask_labelled = measure.label(gt_mask)

        # Obtain GT masks with metastases greater than 6 mm and smaller than 6mm
        gt_mask_small = np.zeros(gt_mask.shape)
        gt_mask_large = np.zeros(gt_mask.shape)
        for i in range(1, np.max(gt_mask_labelled)+1):
            met = gt_mask_labelled == i
            met = np.multiply(met, 1)
            if option1:
                props = measure.regionprops(met)
                maj_ax_le = props[0].major_axis_length
            else:  # Options 2 and 3
                # Common part to both
                met_proj = np.sum(met, axis=2)
                met_proj = met_proj > 0
                met_proj = np.multiply(met_proj, 1)
                if option2:
                    props = measure.regionprops(met_proj)
                    maj_ax_le = props[0].major_axis_length
                else:  # option 3
                    loc = np.nonzero(met_proj)
                    xmin = np.min(loc[0])
                    xmax = np.max(loc[0])
                    ymin = np.min(loc[1])
                    ymax = np.max(loc[1])
                    width1 = xmax - xmin
                    width2 = ymax - ymin
                    if width1 > width2:
                        maj_ax_le = width1
                    else:
                        maj_ax_le = width2
            if maj_ax_le > 6:
                gt_mask_large = gt_mask_large + met
            else:
                gt_mask_small = gt_mask_small + met

        # Load predicted mask
        datafile = os.path.join(pth_pred, patient_name)
        pred_mask = np.squeeze(np.load(datafile))  # probability map
        pred_mask = np.multiply(pred_mask > 0.4, 1)
        # Load bounding box mask
        datafile = os.path.join(pth_bb_masks_tp, patient_name)     ##### ONLY THE TP METASTASES
        bb_mask = np.load(datafile)
        pred_mask = np.multiply(bb_mask, pred_mask)
        pred_mask_labelled = measure.label(pred_mask)

        # Obtain predicted masks with metastases greater than 6 mm and smaller than 6mm
        pred_mask_small = np.zeros(pred_mask.shape)
        pred_mask_large = np.zeros(pred_mask.shape)
        for i in range(1, np.max(pred_mask_labelled)+1):
            met = pred_mask_labelled == i
            met = np.multiply(met, 1)
            if option1:
                props = measure.regionprops(met)
                maj_ax_le = props[0].major_axis_length
            else:  # Options 2 and 3
                # Common part to both
                met_proj = np.sum(met, axis=2)
                met_proj = met_proj > 0
                met_proj = np.multiply(met_proj, 1)
                if option2:
                    props = measure.regionprops(met_proj)
                    maj_ax_le = props[0].major_axis_length
                else:  # option 3
                    loc = np.nonzero(met_proj)
                    xmin = np.min(loc[0])
                    xmax = np.max(loc[0])
                    ymin = np.min(loc[1])
                    ymax = np.max(loc[1])
                    width1 = xmax - xmin
                    width2 = ymax - ymin
                    if width1 > width2:
                        maj_ax_le = width1
                    else:
                        maj_ax_le = width2
            if maj_ax_le > 6:
                pred_mask_large = pred_mask_large + met
            else:
                pred_mask_small = pred_mask_small + met

        val_large = np.sum(gt_mask_large) + np.sum(pred_mask_large)  # Check that at least there is a ground truth metastasis or prediction (check not both masks are 0)
        val_small = np.sum(gt_mask_small) + np.sum(pred_mask_small)  # Check that at least there is a ground truth metastasis or prediction (check not both masks are 0)
        if val_large > 0:
            count_large = count_large + 1
        if val_small > 0:
            count_small = count_small + 1

        # Compute DICE
        if compute_dice:
            # Compute dice for large metastases
            if val_large > 0:
                dice_value = compute_dice_patient(pred_mask_large, gt_mask_large)
                dice_all_large = dice_all_large + dice_value
                print("Large dice value", dice_value)
            # Compute dice for small metastases
            if val_small > 0:
                dice_value = compute_dice_patient(pred_mask_small, gt_mask_small)
                dice_all_small = dice_all_small + dice_value
                print("Small dice value", dice_value)

        # Compute IoU
        if compute_iou:
            # Compute iou for large metastases
            if val_large > 0:
                iou_value = compute_iou_patient(pred_mask_large, gt_mask_large)
                iou_all_large = iou_all_large + iou_value
                print("Large iou value", iou_value)

            # Compute iou for small metastases
            if val_small > 0:
                iou_value = compute_iou_patient(pred_mask_small, gt_mask_small)
                iou_all_small = iou_all_small + iou_value
                print("Small iou value", iou_value)


    # Display values
    if compute_dice:
        dice_all_large = dice_all_large/count_large
        dice_all_small = dice_all_small / count_small
        print("The DICE value (large) is...", dice_all_large)
        print("The DICE value (small) is...", dice_all_small)
    if compute_iou:
        iou_all_large = iou_all_large / count_large
        iou_all_small = iou_all_small / count_small
        print("The IoU value (large) is...", iou_all_large)
        print("The IoU value (small) is...", iou_all_small)





# Notes

# Find slices that contain metastases
# loc = np.sum(np.sum(gt_mask, 0), 0)
# loc = np.nonzero(loc)
# print(loc)
#plt.imshow(gt_mask[:, :, 170])
#plt.show()
