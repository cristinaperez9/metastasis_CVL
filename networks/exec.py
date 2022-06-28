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

########################################################################################################################
# Adapted by: Cristina Almagro Perez, 2022. ETH University
# The new version includes monitoring of the DICE coefficient and saves images of the evolution of training in terms of
# the total loss, the five losses present in Mask R-CNN and detection metrics.
########################################################################################################################

"""execution script."""

import argparse
import os
import time
import torch
import pickle
import numpy as np

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from evaluator import compute_dice_batch
from plotting import plot_batch_prediction
import matplotlib.pyplot as plt
import torch.nn as nn



def train(logger,fold):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    net = model.net(cf, logger).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if cf.resume_to_checkpoint:
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer,cf)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))
        dice_all_epochs = utils.load_dice(cf.fold_dir)

    else:
        dice_all_epochs = np.zeros(cf.num_epochs)

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []

        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')

                dice_epoch = 0
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    #### OBTAIN DICE BATCH VAL ####
                    if cf.evaluate_seg:
                        gt_masks = batch['seg']
                        shape_masks = gt_masks.shape
                        pred_masks = results_dict['seg_preds']
                        pred_masks = np.multiply(pred_masks > 0, 1)  # In case of probabilistic output
                        dice_batch = compute_dice_batch(gt_masks, pred_masks)
                        dice_epoch = dice_epoch + dice_batch
                    #################################
                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])  # just the losses
                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                print("Saving model")
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)
                if cf.evaluate_seg:
                    dice_epoch = dice_epoch / batch_gen['n_val']
                    print("Dice value of current epoch: ", dice_epoch)
                    dice_all_epochs[epoch-1] = dice_epoch

            # update monitoring and prediction plots
            TrainingPlot.update_and_save(monitor_metrics, epoch, fold)
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('plotting predictions from validation sampling.')
            #plot_batch_prediction(batch, results_dict, cf)

        # Save and plot DICE coefficient value
        if cf.evaluate_seg:
            print(cf.fold_dir)
            datafile = os.path.join(cf.fold_dir, 'dice_per_epoch.npy')
            np.save(datafile, dice_all_epochs)
            fig = plt.figure(figsize=(10, 10))
            fig.ax1 = plt.subplot(111)
            fig.ax1.plot(range(1, epoch+1), dice_all_epochs[0:epoch])
            fig.ax1.set_ylim(0, 1)
            fig.ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
            fig.ax1.set_xlabel('epochs')
            fig.ax1.set_ylabel('Dice coefficient')
            fig.ax1.set_xticks(np.arange(1, epoch, 5))
            fig.ax1.grid()
            nm_figure = os.path.join(cf.fold_dir, 'dice_per.epoch.png') #the name of the figure always the same so it will overwrite it
            print("Writing figure with name...", nm_figure)
            fig.savefig(nm_figure)

def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    print("The mAP threshold value is...", cf.ap_match_ious)
    print("The model is...", cf.model)
    print("The architecture is...", cf.res_architecture)

    net = model.net(cf, logger).cuda()
    print(net)
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, net, return_results=True)  # It generates the
    # raw_pred_boxes_hold_out_list.pickle in each folder fold_0, fold_1,...
    # net variable added by me
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config.")
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    folds = args.folds
    resume_to_checkpoint = args.resume_to_checkpoint

    if args.mode == 'train' or args.mode == 'train_test':

        # print(args.exp_source)
        # print(args.exp_dir)
        # print(args.server_env)
        # print(args.use_stored_settings)

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
        if args.dev:
            folds = [0,1]
            cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 if cf.dim==2 else 1, 1, 0, 1
            cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
            cf.test_n_epochs =  cf.save_n_models
            cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)  # This generates numbers 0 1 2 3 4
        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold = fold
            cf.resume_to_checkpoint = resume_to_checkpoint
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            logger = utils.get_logger(cf.fold_dir)
            train(logger,fold) # FOLD ADDED BY ME
            cf.resume_to_checkpoint = None
            if args.mode == 'train_test':
                test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    elif args.mode == 'test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        if args.dev:
            folds = [0,1]
            cf.test_n_epochs = 1; cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        print("The model path is...", cf.model_path)
        # SPECIFY MODEL PATH MANUALLY IF IT DOES NOT WORK OTHERWISE:
        #cf.model_path = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/models/mrcnn.py'
        #cf.model_path = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/models/retina_unet.py'
        cf.model_path = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/models/detection_unet.py'
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            logger = utils.get_logger(cf.fold_dir)
            cf.fold = fold
            test(logger)

            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    elif args.mode == 'analysis':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)

        if cf.hold_out_test_set:
            cf.folds = args.folds
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions(apply_wbc=True)
            utils.create_csv_output(results_list, cf, logger)
            utils.merge_pred_seg(cf, logger)

        else:
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                results_list = predictor.load_saved_predictions(apply_wbc=True)
                logger.info('starting evaluation...')
                evaluator = Evaluator(cf, logger, mode='test')
                evaluator.evaluate_predictions(results_list)
                evaluator.score_test_df()

    # create experiment folder and copy scripts without starting job.
    # useful for cloud deployment where configs might change before job actually runs.
    elif args.mode == 'create_exp':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)
        logger.info('created experiment directory at {}'.format(args.exp_dir))

    else:
        raise RuntimeError('mode specified in args is not implemented...')
