#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import matplotlib
print(matplotlib.__version__)
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess
import sys
ROOT_DIR = os.path.abspath(r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/')
sys.path.append(ROOT_DIR)  # To find local version of the library
import utils.dataloader_utils as dutils
#
# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates


def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    all_data = load_dataset(cf, logger)
    all_pids_list = np.unique([v['pid'] for (k, v) in all_data.items()])

    assert cf.n_train_val_data <= len(all_pids_list), \
        "requested {} train val samples, but dataset only has {} train val samples.".format(
            cf.n_train_val_data, len(all_pids_list))
    train_pids = all_pids_list[:int(2 * cf.n_train_val_data // 3)]
    val_pids = all_pids_list[int(np.ceil(2 * cf.n_train_val_data // 3)):cf.n_train_val_data]

    train_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in train_pids)}
    val_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in val_pids)}

    logger.info("data set loaded with: {} train / {} val patients".format(len(train_pids), len(val_pids)))
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=False)
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(val_data, cf=cf)
        batch_gen['n_val'] = len(val_pids) if cf.max_val_patients is None else min(len(val_pids), cf.max_val_patients)
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen


def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    if cf.hold_out_test_set:
        pp_name = cf.pp_test_name
        test_ix = None
    else:
        pp_name = None
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            fold_list = pickle.load(handle)
        _, _, test_ix, _ = fold_list[cf.fold]
        # warnings.warn('WARNING: using validation set for testing!!!')

    test_data = load_dataset(cf, logger, test_ix, pp_data_path=cf.pp_test_data_path, pp_name=pp_name)
    logger.info("data set loaded with: {} test patients from {}".format(len(test_data.keys()), cf.pp_test_data_path))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(test_data, cf=cf)
    batch_gen['n_test'] = len(test_data.keys()) if cf.max_test_patients=="all" else \
        min(cf.max_test_patients, len(test_data.keys()))

    return batch_gen



def load_dataset(cf, logger, subset_ixs=None, pp_data_path=None, pp_name=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """
    if pp_data_path is None:
        pp_data_path = cf.pp_data_path
    if pp_name is None:
        pp_name = cf.pp_name
    if cf.server_env:
        copy_data = True
        target_dir = os.path.join(cf.data_dest, pp_name)
        if not os.path.exists(target_dir):
            cf.data_source_dir = pp_data_path
            os.makedirs(target_dir)
            subprocess.call('rsync -av {} {}'.format(
                os.path.join(cf.data_source_dir, cf.input_df_name), os.path.join(target_dir, cf.input_df_name)), shell=True)
            logger.info('created target dir and info df at {}'.format(os.path.join(target_dir, cf.input_df_name)))

        elif subset_ixs is None:
            copy_data = False

        pp_data_path = target_dir

    print("The preprocessed data path is...", pp_data_path)

    p_df = pd.read_pickle(os.path.join(pp_data_path, cf.input_df_name))


    if subset_ixs is not None:
        subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
        p_df = p_df[p_df.pid.isin(subset_pids)]
        logger.info('subset: selected {} instances from df'.format(len(p_df)))

    if cf.server_env:
        if copy_data:
            copy_and_unpack_data(logger, p_df.pid.tolist(), cf.fold_dir, cf.data_source_dir, target_dir)

    class_targets = p_df['class_id'].tolist()
    pids = p_df.pid.tolist()

    imgs = [os.path.join(pp_data_path, '{}.npy'.format(pid)) for pid in pids]  # list
    segs = [os.path.join(pp_data_path, '{}.npy'.format(pid)) for pid in pids]   # list

    data = OrderedDict()
    for ix, pid in enumerate(pids):
        data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': [class_targets[ix]]}

    return data



def create_data_gen_pipeline(patient_data, cf, do_aug=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=np.arange(2, cf.dim+2, 1))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=cf.class_specific_seg_flag))
    all_transforms = Compose(my_transforms)
    # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, data, batch_size, cf):
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf

    def generate_train_batch(self):

        batch_data, batch_segs, batch_pids, batch_targets = [], [], [], []
        class_targets_list =  [v['class_target'] for (k, v) in self._data.items()]

        #samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio "batch_sample_slack).
        batch_ixs = dutils.get_class_balanced_patients(
            class_targets_list, self.batch_size, self.cf.head_classes - 1, slack_factor=self.cf.batch_sample_slack)
        patients = list(self._data.items())

        for b in batch_ixs:

            patient = patients[b][1]
            all_data = np.load(patient['data'], mmap_mode='r')
            data = all_data[0]
            seg = all_data[1].astype('uint8')
            batch_pids.append(patient['pid'])
            batch_targets.append(patient['class_target'])
            batch_data.append(data[np.newaxis])
            batch_segs.append(seg[np.newaxis])

        data = np.array(batch_data)
        seg = np.array(batch_segs).astype(np.uint8)
        class_target = np.array(batch_targets)
        return {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}



class PatientBatchIterator(SlimDataLoaderBase):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
    if willing to accept speed-loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """
    def __init__(self, data, cf): #threads in augmenter
        super(PatientBatchIterator, self).__init__(data, 0)
        self.cf = cf
        self.patient_ix = 0
        self.dataset_pids = [v['pid'] for (k, v) in data.items()]
        self.patch_size = cf.patch_size
        if len(self.patch_size) == 2:
            self.patch_size = self.patch_size + [1]


    def generate_train_batch(self):

        pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]
        all_data = np.load(patient['data'], mmap_mode='r')
        data = all_data[0]
        seg = all_data[1].astype('uint8')
        batch_class_targets = np.array([patient['class_target']])

        out_data = data[None, None]
        out_seg = seg[None, None]

        print('check patient data loader', out_data.shape, out_seg.shape)
        batch_2D = {'data': out_data, 'seg': out_seg, 'class_target': batch_class_targets, 'pid': pid}
        converter = ConvertSegToBoundingBoxCoordinates(dim=2, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
        batch_2D = converter(**batch_2D)

        batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                         'patient_roi_labels': batch_2D['roi_labels'],
                         'original_img_shape': out_data.shape})

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return batch_2D



def copy_and_unpack_data(logger, pids, fold_dir, source_dir, target_dir):


    start_time = time.time()
    with open(os.path.join(fold_dir, 'file_list.txt'), 'w') as handle:
        for pid in pids:
            handle.write('{}.npy\n'.format(pid))

    subprocess.call('rsync -av --files-from {} {} {}'.format(os.path.join(fold_dir, 'file_list.txt'),
        source_dir, target_dir), shell=True)
    # dutils.unpack_dataset(target_dir)
    copied_files = os.listdir(target_dir)
    logger.info("copying and unpacking data set finsihed : {} files in target dir: {}. took {} sec".format(
        len(copied_files), target_dir, np.round(time.time() - start_time, 0)))

# if __name__=="__main__":
#     import utils.exp_utils as utils
#     print("Utils module imported correctly")
#     cf_file = utils.import_module("cf", "configs.py")
#     print("Configuration file imported correctly")
#     total_stime = time.time()
#
#
#     cf = cf_file.configs(server_env=False)
#     print("Cf modified correctly")
#     cf.server_env = False
#     logger = utils.get_logger(".")
#     batch_gen = get_test_generator(cf, logger)    # It is a dictionary
#     print(batch_gen)
#     test_batch = next(batch_gen["test"])





if __name__=="__main__":
    import utils.exp_utils as utils
    print("Utils module imported correctly")
    cf_file = utils.import_module("cf", "configs.py")
    print("Configuration file imported correctly")
    total_stime = time.time()


    cf = cf_file.configs(server_env=False)
    print("Cf modified correctly")
    cf.server_env = False
    logger = utils.get_logger(".")
    batch_gen = get_train_generators(cf, logger)    # It is a dictionary
    print(batch_gen)
    train_batch = next(batch_gen["val_sampling"])

    # mydata = train_batch['data']
    # print(mydata.shape)
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(mydata[2, 0, :, :])
    # plt.show()

    # mydata = train_batch['roi_masks']
    # print(mydata.shape)
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(mydata[1, 0, 0, :, :])
    # plt.show()
    # print(train_batch.items())

    # mydata = train_batch['bb_target']
    # print(mydata.shape)
    # print(mydata)
    # print(train_batch.items())

    mydata = train_batch['seg']
    print(mydata.shape)
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(mydata[1, 0, :, :])
    # plt.show()
    #print(mydata)
    #print(train_batch.items())

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))