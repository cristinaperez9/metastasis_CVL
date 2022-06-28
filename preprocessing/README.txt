
Project: Segmenting brain metastases with minimal annotations for radiotherapy applications.
Student: Cristina Almagro Perez
Date: 28/06/2022

################################################################################################################################################################################

Folder preprocessing. It contains the algorithms for image preprocessing: bias correction, statistical analysis of the metastases and bounding box definition

- correct_bias.py. It implements N4 algorithm for bias correction of MR images.

- metastasis_size.py : calculate the size of the metastases in the dataset. It implements three ways of calculating the size based on previous articles:
option1:  measuring the largest diameter in 3D
option2 = project in the craniocaudal direction and measure the largest diameter
option3 = project in the craniocaudal direction and measure the largest cross-sectional dimension (Used in MetNet paper)

- data_statistics.py. It implements the following:
Calculation of the number of patients of each primary cancer type.
Calculation of the distribution of metastases based on size.
Calculation of the distribution of metastases per patient.


################################################################################################################################################################################
