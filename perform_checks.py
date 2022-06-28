
# This code is intended to get familiarized with medical detection toolkit, which is the implementation used in this
# project. It is intended to understand the output of training, inference and analysis and check that all works properly.

#######################################################################################################################
# Cristina Almagro Perez, 2022. ETH University
#######################################################################################################################

# SPECIFY THE FOLLOWING:

gpu_check = True  # Check that code is run on GPU
visualize_npy = True  # Load and visualize an image saved in .npy format
meta_info = True  # Visualize what is stored in the files meta_info0.pickle (meta information of the dataset)
info_df = True  # Visualize what is stored in the files meta_info0.pickle
training_output = True  # Check files generated after training
model_parameters = True  # Check model weights and parameters
inference_output = True  # Check files generated after inference
consolidation_bb = True  # Visualize the output after consolidation of bounding boxes
consolidation_mask = True # Visualize the output after consolidation of segmentation masks (from multiple epochs, folders and patches)
image_and_bb = True  # Visualize and image with its bounding box (Understand toy dataset in Medicaldetectiontoolkit)
preprocessing = True  # Check that images have been preprocessed correctly and masks have been correctly created
training_dictionaries = True  # Visualize dictionaries with the training loss and evaluation metrics
check_inference = True  # Check that inference is correctly performed before performing inference in all test patients
final_detection_results = True  # Analyze the final bounding boxes of detected metastases
figure_methods = True  # Obtain the feature maps of ResNet to include in the figure of the report explaining the
# methods of my project

###########################################################################################################
# Check that code is run on GPU
###########################################################################################################
if gpu_check:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("working on gpu")
    else:
        device = torch.device("cpu")
        print("working on cpu")

###########################################################################################################
# Load and visualize an image saved in .npy format
###########################################################################################################
if visualize_npy:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/train'
    datafile = os.path.join(pth, '0.npy')
    data = np.load(datafile)
    print(data.shape)
    plt.imshow(data[1, :, :])
    plt.show()

############################################################################################################
# Visualize what is stored in the files meta_info0.pickle
############################################################################################################
if meta_info:
    import os
    import pandas as pd
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/train'
    datafile = os.path.join(pth, 'meta_info_100.pickle')
    my_pd = pd.read_pickle(datafile)
    print(my_pd)

############################################################################################################
# Visualize what is stored in info_df.pickle
############################################################################################################
if info_df:
    import os
    import pandas as pd
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/test'
    datafile = os.path.join(pth, 'info_df.pickle')
    my_pd = pd.read_pickle(datafile)
    print(my_pd)
    first = my_pd['path']
    print(first[0])
    print(len(my_pd))
    print(my_pd)

    # It has 1500 rows (as many rows as number of training images)
    # Dimensions 1500 x 3
    # first colum path
    # second column class_id (The class_id are 0 and 1, in my case I think they will be all the same, just not include this parameter)
    # third column is the patient id
    # Note that in this pkl file the images are not ordered (probably they are trained in this order)

    import os
    import pandas as pd
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test1/'#images/'
    datafile = os.path.join(pth, 'meta_info_25.pickle')
    my_pd = pd.read_pickle(datafile)
    print(my_pd)

##################################################################################################################
# Check files generated after training
##################################################################################################################
if training_output:
    import os
    import pandas as pd
    import numpy as np
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/fold_4/last_checkpoint/'
    #pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/fold_4/last_checkpoint/'
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/'
    datafile = os.path.join(pth, 'datafile_fold_4epoch_1.pickle')
    myfile = pd.read_pickle(datafile)
    #print(myfile)
    train_values = myfile['train']  # Ordered dictionary
    print(train_values.keys())  # foreground_ap, patient_ap,patient_auc, monitor_values
    val_values = myfile['val']  # Ordered dictionary
    print(val_values.keys())  # foreground_ap, patient_ap,patient_auc, monitor_values
    # # train_losses = train_values['monitor_values']
    # # print(type(train_losses))
    print(val_values['malignant_ap'])
###################################################################################################################
# Load the file containing the parameters of the model
###################################################################################################################
if model_parameters:
    import torch
    import os
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/fold_/last_checkpoint/'
    datafile = os.path.join(pth, 'params.pth')
    model = torch.load(datafile, map_location=torch.device('cpu'))
    #print(model)
    print(model.keys())
    opt = model['optimizer']
    print(opt)
    opt = model['state_dict']
    a = list(opt)
    print(a)
    for kk in range(len(opt.keys())):
        key_name = a[kk]
        key_name_new = key_name[0].capitalize() + key_name[1:]
        opt[key_name_new] = opt.pop(key_name)
    print(opt.keys())

###################################################################################################################
# Check .pkl files generated after inference
###################################################################################################################
if inference_output:
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/'
    datafile = os.path.join(pth, '3_map_0.1_test_df.pickle')
    myfile = pd.read_pickle(datafile)
    myfile = myfile[myfile["match_iou"] == 0.1]
    myfile = myfile[myfile["pid"] == str(30)]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
         print(myfile)

    # Check output segmentation mask of Toydataset provided by medical detection toolkit
    datafile = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/test/98.npy'
    data = np.load(datafile)
    plt.imshow(data[0, :, :])
    plt.show()
    plt.imshow(data[0, :, :])
    plt.imshow(data[0, :, :], interpolation=None, alpha=0.25)
    plt.show()
####################################################################################################################
# Load .csv file created after consolidation of the bounding boxes
####################################################################################################################
if consolidation_bb:
    import pandas as pd
    import numpy as np
    import os
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/'
    datafile = os.path.join(pth, 'results_hold_out.csv')
    myfile = pd.read_csv(datafile)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
         print(myfile)
    pid = myfile['patientID']
    pid = pid.to_numpy()
    unique_pid = np.unique(pid)
    print(len(unique_pid))

######################################################################################################################
# Load an image and visualize predicted bounding box
######################################################################################################################
if image_and_bb:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/test'
    datafile = os.path.join(pth, 'info_df.pickle')
    my_pd = pd.read_pickle(datafile)
    first = my_pd['path']
    print(first[0])
    print(len(my_pd))
    print(my_pd)

    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/test'
    datafile = os.path.join(pth, '484.npy')
    data = np.load(datafile) #image
    print(data.shape)
    img = data[0, :, :]
    # plt.imshow(img)
    # plt.show()

    # load bounding box results and draw bounding box on top of the image

    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/donuts_shape/'
    datafile = os.path.join(pth, 'results_hold_out.csv')
    myfile = pd.read_csv(datafile)
    entries_patient = myfile.loc[myfile['patientID'] == 484]
    print(entries_patient)
    #
    # # Since the second entry has a higher score, I select the second row
    entry_patient = entries_patient.loc[1]
    #print(entry_patient)
    coords = entry_patient['coords']  # str
    coords = [280, 185, 318, 227]
    box = np.zeros(img.shape)
    box[coords[0]:coords[2], coords[1]:coords[3]] = 1
    plt.imshow(img)
    plt.imshow(box, alpha=0.2)
    plt.show()

#######################################################################################################
# Load pickle my dataset
#######################################################################################################
if meta_info:
    import numpy
    import pandas as pd
    import os
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test1/'
    #datafile = os.path.join(pth, 'meta_info_50.pickle')
    datafile = os.path.join(pth, 'info_df.pickle')
    myfile = pd.read_pickle(datafile)
    myfile = myfile.iloc[1]
    print(myfile)
    myfile = myfile.iloc[0]
    print(myfile.loc['path_image'])

#########################################################################################################
# Check preprocess images & masks
#########################################################################################################
if preprocessing:
    import os
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test/'
    datafile = os.path.join(pth, '50.npy')
    myfile = np.load(datafile)
    print(myfile.shape)
    img = myfile[0, :, :, :]
    seg = myfile[1, :, :, :]
    print(img.shape)
    print(seg.shape)

    # Visualize random section to check normalization: normalization performed correctly.
    plt.imshow(img[:, :, 80], cmap='gray')
    plt.show()
    print(np.max(img))
    print(np.min(img))

    # Check mask
    plt.imshow(seg[:, :, 88], cmap='gray')
    plt.show()
    print(np.max(seg))
    print(np.min(seg))

    # Check first and second slice (first and second slides were duplicated as part of the preprocessing process)
    a = np.array_equal(img[:,:,0], img[:,:,1])
    b = np.array_equal(img[:,:,271], img[:,:,270])
    c = np.array_equal(seg[:,:,0], seg[:,:,1])
    d = np.array_equal(seg[:,:,271], seg[:,:,270])
    print(a)
    print(b)
    print(c)
    print(d)

#######################################################################################################################
# Analyze variables for plotting
#######################################################################################################################
if training_dictionaries:
    import pandas as pd
    import os

    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/'
    datafile = os.path.join(pth, 'datafile.pickle')  # It is a dictionary
    mydf = pd.read_pickle(datafile)
    print(mydf)
    print(mydf.keys())  # It has the keys 'train' & 'val'

    # Training
    mydf_train = mydf['train']  # It is an ordered dict
    print(mydf_train)
    print(mydf_train.keys())  # It has the keys foreground_ap, patient_ap, patient_auc, monitor_values

    # Validation
    mydf_val = mydf['val']  # It is an ordered dict
    print(mydf_val.keys())  # It has the keys foreground_ap, patient_ap, patient_auc, monitor_values
    print(mydf_val)
    monitor_metrics_val = mydf_val['monitor_values']
    print(type(monitor_metrics_val))  # list
    val0 = monitor_metrics_val[0]  # [] it is empty
    val1 = monitor_metrics_val[1]  # it contains all the losses
    print(type(val1))  # list
    val1_0 = val1[0]
    print(val1_0)  # It contains the loss & class loss of the
    print(len(val1))

#################################################################################################################
# Create test folder with 2 patients to check if inference works
#################################################################################################################
if check_inference:
    import os
    import pandas as pd
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/test0/'
    datafile = os.path.join(pth, 'info_df.pickle')
    my_pd = pd.read_pickle(datafile)
    print(my_pd)
    myfile = my_pd.iloc[50:]
    print(myfile)
    import pickle
    with open(os.path.join(pth, 'info_df3.pickle'), 'wb') as handle:
        pickle.dump(myfile, handle)

#################################################################################################################
# Analyze pickle files generated after inference in my dataset
#################################################################################################################
if inference_output:
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/fold_3/'
    datafile = os.path.join(pth, 'raw_pred_boxes_hold_out_list_0.5.pickle')
    myfile = pd.read_pickle(datafile)
    print(myfile)

    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/results_test1'
    datafile = os.path.join(pth, '3_map_0.5_test_df.pickle')
    myfile = pd.read_pickle(datafile)
    with pd.option_context('display.max_rows',None,'display.max_columns',None):
        print(myfile)

    print(len(myfile))  # The length is 2 since I have initialized analyzed 2 patients
    a = myfile[0]  # Analyze first patient # It is a list
    a = a[0][0]
    print(type(a))
    print(len(a))  # 8241 (huge number of bounding boxes?) first patient   # 13305 for the first patient fold 1
    #print(a)

    # box_coords, box_score, box_type, box_pred_class_id, patch_id, box_patch_center_factor, box_n_overlaps
    # 3: box_type: it is always det (It could also be gt which means ground truth) det just means is from a prediction
    # 4: box_pred_class_id : it is always equal to 1
    # 5: patch_id : I guess is just the patch from which the bounding box was extracted
    # 6: box_patch_center_factor:
    # 7: box_n_overlaps: required for consolidation: amount over overlapping patches at the box's position


#############################################################################################################
# Analyze results_hold_out.csv of my dataset
#############################################################################################################
if final_detection_results:
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/'
    datafile = os.path.join(pth, 'results_hold_out.csv')
    myfile = pd.read_csv(datafile)
    print(myfile)
    entries_patient = myfile.loc[myfile['patientID'] == 1]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(entries_patient)

#######################################################################################################################
# Obtain feature maps with ResNet to include in the figure explaining the methods of my project
# #####################################################################################################################
# Import necessary packages:
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
import os
from PIL import Image
import scipy.io as sio

##############################################################################################################
# Auxiliary functions
##############################################################################################################


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img
###############################################################################################################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

# Load MR image
print("====> Loading data")
pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/preprocessed/train/'
datafile = os.path.join(pth, '5.npy')
data = np.load(datafile)
im0 = data[0, :, :, :]

# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#mask = data[1, :, :, :]

# Find location of metastases
# loc = np.sum(np.sum(mask, 0), 0)
# loc = np.nonzero(loc)
# print(loc)

#plt.imshow(im[:, :, 80], cmap='gray')
#plt.imshow(mask[:, :, 80], alpha=0.2)
#plt.show()
print("====> Loading model")
model = models.resnet50(pretrained=True)
print(model)
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []# get all the model children as list
model_children = list(model.children())#counter to keep count of the conv layers
counter = 0#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

vol8 = np.zeros((112,112,272))
vol5 = np.zeros((112,112,272))

for kk in range(im0.shape[2]):
    print("===> Section...", kk)
    im = im0[:, :, kk]
    img = convert(im, 0, 255, np.uint8)
    img = Image.fromarray(img).convert("RGB")  #Image is between 0-255

    image = transform(img)
    #image = img.resize((224, 224))

    #image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)

    # Generate feature maps
    print("====> Generating feature maps")
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))#print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)


    processed = []
    print("====> Processing feature maps")
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

        for fm in processed:
            print(fm.shape)

    print("====> Generating figure")
    #vol8[:, :, kk] = np.resize(processed[7], (384, 384))
    #vol5[:, :, kk] = np.resize(processed[4], (384, 384))
    vol8[:, :, kk] = processed[7]
    vol5[:, :, kk] = processed[4]

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        temp = processed[i]
        temp[temp == 0] = -0.05
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
        #plt.show()
        if i == 19:
            outpth = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/plots_new/'
            plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
sio.savemat('vol8_new.mat', {'vol8': vol8})
sio.savemat('vol5_new.mat', {'vol5': vol8})



