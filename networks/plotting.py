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

###################################################################################################################
# Adapted by: Cristina Almagro Perez, 2022. ETH University
# -----------------------------------------------------------------------------------------------------------------
# Contributions:
# Create plot with the five losses of Mask R-CNN
# Create plot with the total loss and mAP
# Create plot of selected detection metrics
####################################################################################################################

# Import necessary packages
import matplotlib
#matplotlib.use('Agg') # THIS WAS SET IN ORIGINAL CODE
#matplotlib.use('pdf') # ADDED BY ME
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from copy import deepcopy


def plot_batch_prediction(batch, results_dict, cf, outfile= None):
    """
    plot the input images, ground truth annotations, and output predictions of a batch. If 3D batch, plots a 2D projection
    of one randomly sampled element (patient) in the batch. Since plotting all slices of patient volume blows up costs of
    time and space, only a section containing a randomly sampled ground truth annotation is plotted.
    :param batch: dict with keys: 'data' (input image), 'seg' (pixelwise annotations), 'pid'
    :param results_dict: list over batch element. Each element is a list of boxes (prediction and ground truth),
    where every box is a dictionary containing box_coords, box_score and box_type.
    """
    if outfile is None:
        outfile = os.path.join(cf.plot_dir, 'pred_example_{}.png'.format(cf.fold))

    data = batch['data']
    segs = batch['seg']
    pids = batch['pid']
    # for 3D, repeat pid over batch elements.
    if len(set(pids)) == 1:
        pids = [pids] * data.shape[0]

    seg_preds = results_dict['seg_preds']
    roi_results = deepcopy(results_dict['boxes'])

    # Randomly sampled one patient of batch and project data into 2D slices for plotting.
    if cf.dim == 3:
        patient_ix = np.random.choice(data.shape[0])
        data = np.transpose(data[patient_ix], axes=(3, 0, 1, 2))

        # select interesting foreground section to plot.
        gt_boxes = [box['box_coords'] for box in roi_results[patient_ix] if box['box_type'] == 'gt']
        if len(gt_boxes) > 0:
            z_cuts = [np.max((int(gt_boxes[0][4]) - 5, 0)), np.min((int(gt_boxes[0][5]) + 5, data.shape[0]))]
        else:
            z_cuts = [data.shape[0]//2 - 5, int(data.shape[0]//2 + np.min([10, data.shape[0]//2]))]
        p_roi_results = roi_results[patient_ix]
        roi_results = [[] for _ in range(data.shape[0])]

        # iterate over cubes and spread across slices.
        for box in p_roi_results:
            b = box['box_coords']
            # dismiss negative anchor slices.
            slices = np.round(np.unique(np.clip(np.arange(b[4], b[5] + 1), 0, data.shape[0]-1)))
            for s in slices:
                roi_results[int(s)].append(box)
                roi_results[int(s)][-1]['box_coords'] = b[:4]

        roi_results = roi_results[z_cuts[0]: z_cuts[1]]
        data = data[z_cuts[0]: z_cuts[1]]
        segs = np.transpose(segs[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]
        seg_preds = np.transpose(seg_preds[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]
        pids = [pids[patient_ix]] * data.shape[0]

    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 2, 3]:
            assert data.shape[i] == segs.shape[i] == seg_preds.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement!'
                      'Shapes {} vs. {} vs {}'.format(data.shape, segs.shape, seg_preds.shape))


    show_arrays = np.concatenate([data, segs, seg_preds, data[:, 0][:, None]], axis=1).astype(float)
    approx_figshape = (4 * show_arrays.shape[0], 4 * show_arrays.shape[1])
    fig = plt.figure(figsize=approx_figshape)
    gs = gridspec.GridSpec(show_arrays.shape[1] + 1, show_arrays.shape[0])
    gs.update(wspace=0.1, hspace=0.1)
    for b in range(show_arrays.shape[0]):
        for m in range(show_arrays.shape[1]):

            ax = plt.subplot(gs[m, b])
            ax.axis('off')
            if m < show_arrays.shape[1]:
                arr = show_arrays[b, m]

            if m < data.shape[1] or m == show_arrays.shape[1] - 1:
                cmap = 'gray'
                vmin = None
                vmax = None
            else:
                cmap = None
                vmin = 0
                vmax = cf.num_seg_classes - 1

            if m == 0:
                plt.title('{}'.format(pids[b][:10]), fontsize=20)

            plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            if m >= (data.shape[1]):
                for box in roi_results[b]:
                    if box['box_type'] != 'patient_tn_box': # don't plot true negative dummy boxes.
                        coords = box['box_coords']
                        if box['box_type'] == 'det':
                            # dont plot background preds or low confidence boxes.
                            if box['box_pred_class_id'] > 0 and box['box_score'] > 0.1:
                                plot_text = True
                                score = np.max(box['box_score'])
                                score_text = '{}|{:.0f}'.format(box['box_pred_class_id'], score*100)
                                # if prob detection: plot only boxes from correct sampling instance.
                                if 'sample_id' in box.keys() and int(box['sample_id']) != m - data.shape[1] - 2:
                                        continue
                                # if prob detection: plot reconstructed boxes only in corresponding line.
                                if not 'sample_id' in box.keys() and  m != data.shape[1] + 1:
                                    continue

                                score_font_size = 7
                                text_color = 'w'
                                text_x = coords[1] + 10*(box['box_pred_class_id'] -1) #avoid overlap of scores in plot.
                                text_y = coords[2] + 5
                            else:
                                continue
                        elif box['box_type'] == 'gt':
                            plot_text = True
                            score_text = int(box['box_label'])
                            score_font_size = 7
                            text_color = 'r'
                            text_x = coords[1]
                            text_y = coords[0] - 1
                        else:
                            plot_text = False

                        color_var = 'extra_usage' if 'extra_usage' in list(box.keys()) else 'box_type'
                        color = cf.box_color_palette[box[color_var]]
                        plt.plot([coords[1], coords[3]], [coords[0], coords[0]], color=color, linewidth=1, alpha=1) # up
                        plt.plot([coords[1], coords[3]], [coords[2], coords[2]], color=color, linewidth=1, alpha=1) # down
                        plt.plot([coords[1], coords[1]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # left
                        plt.plot([coords[3], coords[3]], [coords[0], coords[2]], color=color, linewidth=1, alpha=1) # right
                        if plot_text:
                            plt.text(text_x, text_y, score_text, fontsize=score_font_size, color=text_color)

    try:
        print("Saving figure. Output path is...", outfile)
        plt.savefig(outfile)
    except:
        raise Warning('failed to save plot.')
    plt.close(fig)



class TrainingPlot_2Panel():


    def __init__(self, cf):

        self.file_name = cf.plot_dir + '/monitor_{}'.format(cf.fold)
        self.exp_name = cf.fold_dir
        self.do_validation = cf.do_validation
        self.separate_values_dict = cf.assign_values_to_extra_figure
        self.figure_list = []
        self.figure_list1 = []
        for n in range(cf.n_monitoring_figures):
            self.figure_list.append(plt.figure(figsize=(10, 10)))  # (10,6)
            self.figure_list[-1].ax1 = plt.subplot(111)
            self.figure_list[-1].ax1.set_xlabel('epochs')
            self.figure_list[-1].ax1.set_ylabel('loss / metrics')
            self.figure_list[-1].ax1.set_xlim(0, cf.num_epochs)
            self.figure_list[-1].ax1.grid()

        self.figure_list[0].ax1.set_ylim(0, 5)  # It was 1.5 before
        self.color_palette1 = ['b', 'c', 'r', 'purple', 'm', 'y', 'k', 'tab:gray']
        self.color_palette = ['b', 'c', 'lightcoral', 'gold', 'yellowgreen','plum','r', 'purple', 'm', 'y', 'k', 'tab:gray']
        self.num_epochs = cf.num_epochs

    color_palette = ['b', 'c', 'lightcoral', 'gold', 'yellowgreen', 'plum', 'k']
    def update_and_save(self, metrics, epoch, fold):

        for figure_ix in range(len(self.figure_list)):
            fig1 = plt.figure(figsize=(10, 10))
            fig1.ax1 = plt.subplot(111)
            fig1.ax1.set_xlabel('epochs')
            fig1.ax1.set_ylabel('loss / metrics')
            fig1.ax1.set_xlim(0, self.num_epochs)
            fig1.ax1.set_xlim(0, 300)
            fig1.ax1.grid()
            fig1.ax1.set_ylim(0, 4)  # It was 1.5 before
            print("Plotting figure 1 ...")
            detection_monitoring_plot(fig1.ax1, metrics, self.exp_name, self.color_palette, epoch, figure_ix,
                                      self.separate_values_dict,
                                      self.do_validation,fold)
            #nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/plots1/'
            nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/resnet101/'
            nm_figure_final = nm_figure + 'all_metrics_' + 'fold_' + str(fold) + 'epoch_' + str(epoch)
            print("Saving figure 1")
            fig1.savefig(nm_figure_final)
            plt.close(fig1)


        for figure_ix in range(len(self.figure_list)):
            fig2 = plt.figure(figsize=(10, 10))
            fig2.ax1 = plt.subplot(111)
            fig2.ax1.set_xlabel('epochs')
            fig2.ax1.set_ylabel('loss')
            #fig2.ax1.set_xlim(0, self.num_epochs)
            fig2.ax1.set_xlim(0, 300)
            fig2.ax1.grid()
            fig2.ax1.set_ylim(0, 4)  # It was 1.5 before
            print("Plotting figure 2 ...")
            detection_monitoring_plot_CP(fig2.ax1, metrics, self.exp_name, self.color_palette, epoch, figure_ix,
                                      self.separate_values_dict,
                                      self.do_validation, fold)
            #nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/plots/'
            #nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/plots1/'
            nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/resnet101/'
            nm_figure_final = nm_figure + 'fold_' + str(fold) + 'epoch_' + str(epoch) + 'only_monitor_metrics'
            fig2.savefig(nm_figure_final)
            print("Saving figure 2")
            plt.close(fig2)

        for figure_ix in range(len(self.figure_list)):
            fig3 = plt.figure(figsize=(10, 10))
            fig3.ax1 = plt.subplot(111)
            fig3.ax1.set_xlabel('epochs')
            fig3.ax1.set_ylabel('Mask-RCNN losses')
            #fig3.ax1.set_xlim(0, self.num_epochs)
            fig3.ax1.set_xlim(0, 300)
            fig3.ax1.grid()
            fig3.ax1.set_ylim(0, 4)  # It was 1.5 before
            color_palette = ['b', 'c', 'lightcoral', 'gold', 'yellowgreen', 'plum', 'k']
            print("Plotting figure 3 ...")
            detection_monitoring_plot_all_losses(fig3.ax1, metrics, self.exp_name, color_palette, epoch, figure_ix,
                                         self.separate_values_dict,
                                         self.do_validation, fold)

            #nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/toy_dataset/plots/'
            #nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/plots1/'
            nm_figure = '/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/resnet101/'
            nm_figure_final = nm_figure + 'all_losses_' + 'fold_' + str(fold) + 'epoch_' + str(epoch) + 'only_monitor_metrics'
            fig3.savefig(nm_figure_final)
            plt.close(fig3)



def detection_monitoring_plot(ax1, metrics, exp_name, color_palette, epoch, figure_ix, separate_values_dict, do_validation,fold):

    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    print("Printing the monitor values...", monitor_values_keys)
    separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    if figure_ix == 0:
        plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
        plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
        #print(metrics['train'].keys())
    else:
        plot_keys = separate_values_dict[figure_ix]

    #print("The monitor values keys are...", monitor_values_keys)
    print("The plot keys are...", plot_keys)
    x = np.arange(1, epoch + 1)

    for kix, pk in enumerate(plot_keys):
        #print(" Printing kix...", kix)
        #print(" Printing pk...", kix)
        if pk in metrics['train'].keys():
            #print("pk is in metrics train, pk is...", pk)
            y_train = metrics['train'][pk][1:]
            ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])

            # Plotting only the metric that I am interested in
            # if kix == 2: #6
            #     y_train = metrics['train'][pk][1:]  # Plot only average precision
            #     ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])  # I MODIFIED THIS
            #
            #     if do_validation:
            #         y_val = metrics['val'][pk][1:]
            #         ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])
            #         # Print average precision metric
            #         ap_val = y_train[-1]
            #         print("The average precision metric for the validation set is...", ap_val)

            if do_validation:
                y_val = metrics['val'][pk][1:]
                ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])  # I MODIFIED THIS
        else:
            if kix == 0: # only plot total training loss
                y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]
                ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])
                if do_validation:
                    y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]
                    ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])
                    # Print average precision metric
                    ap_val = y_train[-1]
                    print("The average precision metric for the validation set is...", ap_val)

        if do_validation:
            ax1.legend(loc='center left')

        if epoch > 1:
            ax1.legend_ = None


def detection_monitoring_plot_CP(ax2, metrics, exp_name, color_palette, epoch, figure_ix, separate_values_dict,
                              do_validation, fold):

    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    if figure_ix == 0:
        plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
        plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
    else:
        plot_keys = separate_values_dict[figure_ix]

    #print("The monitor values keys are...", monitor_values_keys)

    x = np.arange(1, epoch + 1)

    for kix, pk in enumerate(plot_keys):
        #print(" Printing kix...", kix)
        #print(" Printing pk...", kix)
        if pk in metrics['train'].keys():
            #print("pk is in metrics train")
            y_train = metrics['train'][pk][1:]
            if do_validation:
                y_val = metrics['val'][pk][1:]
        else:
            y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]

            if kix == 0:  # only plot total train loss
                ax2.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])

            if do_validation:
                y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]

                if kix == 0:  # only plot total validation loss
                    ax2.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])

        if do_validation:
            ax2.legend(loc='center left')

        if epoch > 1:
            print("Epoch is greater than 1")
            ax2.legend_ = None

def detection_monitoring_plot_all_losses(ax3, metrics, exp_name, color_palette, epoch, figure_ix,
                                 separate_values_dict,
                                 do_validation, fold):

    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    if figure_ix == 0:
        plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
        plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
    else:
        plot_keys = separate_values_dict[figure_ix]

    print("The monitor values keys are...", monitor_values_keys)

    x = np.arange(1, epoch + 1)

    for kix, pk in enumerate(plot_keys):
        print(" Printing kix...", kix)
        print(" Printing pk...", kix)
        if pk in metrics['train'].keys():
            print("pk is in metrics train")
            y_train = metrics['train'][pk][1:]
            if do_validation:
                y_val = metrics['val'][pk][1:]
        else:
            y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]
            ax3.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])

            #if do_validation:
                #y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]
                #ax3.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])

        if do_validation:
            ax3.legend(loc='center left')

        if epoch > 1:
            print("Epoch is greater than 1")
            ax3.legend_ = None


def plot_prediction_hist(label_list, pred_list, type_list, outfile):
    """
    plot histogram of predictions for a specific class.
    :param label_list: list of 1s and 0s specifying whether prediction is a true positive match (1) or a false positive (0).
    False negatives (missed ground truth objects) are artificially added predictions with score 0 and label 1.
    :param pred_list: list of prediction-scores.
    :param type_list: list of prediction-types for stastic-info in title.
    """
    preds = np.array(pred_list)
    labels = np.array(label_list)
    title = outfile.split('/')[-1] + ' count:{}'.format(len(label_list))
    plt.figure()
    plt.yscale('log')
    if 0 in labels:
        plt.hist(preds[labels == 0], alpha=0.3, color='g', range=(0, 1), bins=50, label='false pos.')
    if 1 in labels:
        plt.hist(preds[labels == 1], alpha=0.3, color='b', range=(0, 1), bins=50, label='true pos. (false neg. @ score=0)')

    if type_list is not None:
        fp_count = type_list.count('det_fp')
        fn_count = type_list.count('det_fn')
        tp_count = type_list.count('det_tp')
        pos_count = fn_count + tp_count
        title += ' tp:{} fp:{} fn:{} pos:{}'. format(tp_count, fp_count, fn_count, pos_count)

    plt.legend()
    plt.title(title)
    plt.xlabel('confidence score')
    plt.ylabel('log n')
    plt.savefig(outfile)
    plt.close()


def plot_stat_curves(stats, outfile):

    for c in ['roc', 'prc']:
        plt.figure()
        for s in stats:
            if s[c] is not None:
                plt.plot(s[c][0], s[c][1], label=s['name'] + '_' + c)
        plt.title(outfile.split('/')[-1] + '_' + c)
        plt.legend(loc=3 if c == 'prc' else 4)
        plt.xlabel('precision' if c == 'prc' else '1-spec.')
        plt.ylabel('recall')
        plt.savefig(outfile + '_' + c)
        plt.close()
