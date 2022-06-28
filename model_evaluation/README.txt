
Project: Segmenting brain metastases with minimal annotations for radiotherapy applications.
Student: Cristina Almagro Perez
Date: 28/06/2022

###########################################################################################################################################################################################

Folder model_evaluation. It contains the algorithms employed for the evaluation of detection and segmentation of the predicted brain metastases.

- detection_evaluation.py. Calculate precision, recall, F1-score and accuracy of the model. As input it requires the file 'results_hold_out.csv' generated after analysis
(Medical Detection Toolkit run mode) and the file 'X_test_df.pickle' were X is the folder used during training (one of the following: 0,1,2,3,4). This file is generated during inference 
(Medical Detection Toolkit run mode).

- postprocessing.py. The model only outputs the bounding box coordinates of the detected metastases and the metastasis probability map. This algorithm generates masks of the bounding
boxes from the predicted bounding box coordinates. Finally, during segmentation evaluation, the output probability maps are multiplied by the these masks, so only the detected metastases are included in the segmentation masks and considered for evaluation.

- segmentation_evaluation.py: First, the output probability maps are multiplied by the masks generated after postprocessing, so only the detected metastases are included in the segmentation masks and considered for evaluation. Then the DSC (Dice Similarity Coefficient) and IoU (Intersection over Union) are calculated.

- visualize_detection.py: It creates 2D visualizations of the ground truth and predicted brain metastasis for qualitative evaluation.

- graphs_report.py. It generates the graphs included in the report.


############################################################################################################################################################################################
