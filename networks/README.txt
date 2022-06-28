
Project: Segmenting brain metastases with minimal annotations for radiotherapy applications.
Student: Cristina Almagro Perez
Date: 28/06/2022

#######################################################################################################################################################################

The 3D Mask R-CNN implementation of this project was adapted from Medical Detection Toolkit repository: https://github.com/MIC-DKFZ/medicaldetectiontoolkit
I really advice to carefully read the above repository to understand the framework. The steps to run the algorithm are explained below. However, I still advice to run
the implementation with the toy dataset in order to get familizared with the framework.

The code is included within the folder medicaldetectiontoolkit. IMPORTANT: IT REQUIRES titan_xp OR geforce_gtx_titan_x GPU.

The code is run in three main steps: TRAINING, INFERENCE and ANALYSIS.The file sbatch_test.sh includes the lines required for training the algorithm.
The file sbatch_test_inference.sh includes the lines required for performing inference and the file sbatch_test_analysis.sh the code required for performing analysis.

STEP 0: Data preprocessing. Witihin the folder experiments/my_dataset you can find a preprocess.py file. This file saved the images as the format required by 
Medical Detection Toolkit. It normalizes images between 0 and 1, splits the images intro training and testing and saves the MR images together with the masks in a single
variable .npy. It also creates metadata information with the name of each patient and the path for each patient.
The output of this algorithm is for example '1.npy' for patient 1. This variable has dimensions 2 x 384 x 384 x 272 (The first channel is the MR image and the second
channel the corresponding binary mask)


STEP 1: read the config.py file and modify the aspects that require appropiate (in principle only the root folder is required to be changed).

STEP 2: network training. By default it performs five-fold cross-validation. It will save the five-best epochs according to the mean average precision of the validation set.
To run the algorithm follow the lines contained in sbatch_test.sh.

STEP 3: network inference. Inference is performed in patches of dimensions 128 x 128 x 64. Then, they are consolidated to the world coordinates. This step generates
the raw predictions prior to consolidation. The output of this part is the following:
- raw_pred_boxes_hold_out_list_0.1.pickle (It contains the raw bounding boces prior to consolidation).
- raw_pred_boxes_&_seg_hold_out_list_0.1.pickle (It contains the raw segmentations prior to consolidation).
Besides it generates two other files:
- X_test_df.pickle : it contains a dictionary. For each patient (in the test folder), there will be as many rows as raw predictions. For each predicition, there is an 
entry named 'pred_score' which contains the confidence of the predicition. Besides, there is an entry indicating the type of predicition (True positive, False positive 
or False negative)
- results_ap_0.1 : mean average precision results in the test folder.
To run network inference follow the lines in sbatch_test_inference.sh

STEP 4: analysis. It performs consolidation of predictions (bounding boxes and segmentation masks). Bounding box consolidation is performed through box clustering algorithm
and segmentation consolidation is performed through averaging.
The output of this step are:
- results_hold_out.csv (Each row is a metatasis prediction. The columns are the patient ID, the bounding box coordinates and the confidence of the prediction.
THESE ARE THE FINAL METATASES DETECTION.
- It creates a folder named test_pred_seg. In this folder it saves the predicted metastasis probabiliy map for each patient.
To run network analysis follow the lines in sbatch_test_analysis.sh



Finally, the perform_checks.py code includes visualization of the outputs of the training/inference/analysis steps to ensure in each step that everything was run properly.


#######################################################################################################################################################################
