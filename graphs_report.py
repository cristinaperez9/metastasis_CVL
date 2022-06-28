##########################################################################################################
# Cristina Almagro Perez, 2022. ETH University.
##########################################################################################################

# Create graphs for report.

# Import necessary packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SPECIFY THIS:
plot_1 = True  # total training loss + (total validation loss) + training mAP + (validation mAP)
plot_2 = False  # Plot five losses of Mask R-CNN
precision_recall = False  # precision - recall curve
probability_map = False  # obtain threshold output metastases probability maps
f1_score = True  # Plot F1-score vs. confidence threshold
fold = 1  # Specify: 0, 1, 2, 3, 4
pth = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/fold_' + str(fold) + '/'
# pth = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures/fold_' + str(fold) + '/'


########################################################################################################################
# Auxiliary functions
########################################################################################################################

def detection_monitoring_plot(ax1, metrics, color_palette, epoch, figure_ix, separate_values_dict, do_validation):
    ''' Function for plotting the total training/validation loss and the training mAP/validation mAP '''
    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    if figure_ix == 0:
        plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
        plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
    else:
        plot_keys = separate_values_dict[figure_ix]

    x = np.arange(1, epoch + 1)
    print("Plot_keys are...", plot_keys)
    for kix, pk in enumerate(plot_keys):

        if pk in metrics['train'].keys():

            # Plotting only the metric that I am interested in
            if kix == 6:  # 2 if not monitor all losses
                y_train = metrics['train'][pk][1:epoch+1]  # Plot only average precision

                ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])  # I MODIFIED THIS

                if do_validation:
                    y_val = metrics['val'][pk][1:epoch+1]
                    ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])
                    # Print average precision metric
                    ap_val = y_train[-1]
                    print("The average precision metric for the validation set is...", ap_val)

        else:
            if kix == 0:  # only plot total training loss
                y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]
                print(len(y_train))
                ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])
                if do_validation:
                    y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]
                    ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])


def detection_monitoring_plot_all_losses(ax3, metrics, color_palette, epoch, figure_ix, separate_values_dict, do_validation):

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

        if do_validation:
            ax3.legend(loc='center left')

        if epoch > 1:
            ax3.legend_ = None

#################################################################################################################
# Plot: total training loss + (total validation loss) + training mAP + (validation mAP)
#################################################################################################################


if plot_1:

    # Load metrics to plot
    datafile = os.path.join(pth, 'last_checkpoint/monitor_metrics.pickle')
    metrics = pd.read_pickle(datafile)
    epoch = 378  #300
    do_validation = True

    fig1 = plt.figure(figsize=(10, 10))
    fig1.ax1 = plt.subplot(111)
    fig1.ax1.set_xlabel('epochs')
    fig1.ax1.set_ylabel('loss / metrics')
    fig1.ax1.set_xlim(0, 400)  # It was 300
    fig1.ax1.grid()
    fig1.ax1.set_ylim(0, 4)
    color_palette = ['b', 'c', 'r', 'purple', 'm', 'y', 'k', 'tab:gray']
    separate_values_dict = {}

    nm_figure = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/graphs_report/'
    nm_figure_final = nm_figure + 'train_loss_ap_' + 'fold_' + str(fold) + '_epoch_' + str(epoch) + 'ResNet101'

    detection_monitoring_plot(fig1.ax1, metrics, color_palette, epoch, 0, separate_values_dict, do_validation)
    fig1.savefig(nm_figure_final)

#######################################################################################################################
# Plot 5 losses of Mask R-CNN vs. epoch
#######################################################################################################################

if plot_2:
    # Load metrics to plot
    datafile = os.path.join(pth, 'last_checkpoint/monitor_metrics.pickle')
    metrics = pd.read_pickle(datafile)
    epoch = 150
    do_validation = False

    fig2 = plt.figure(figsize=(10, 10))
    fig2.ax1 = plt.subplot(111)
    fig2.ax1.set_xlabel('epochs')
    fig2.ax1.set_ylabel('loss')
    fig2.ax1.set_xlim(0, 150)
    fig2.ax1.grid()
    fig2.ax1.set_ylim(0, 4)
    color_palette = ['b', 'c', 'lightcoral', 'gold', 'yellowgreen','plum','r', 'purple', 'm', 'y', 'k', 'tab:gray']
    separate_values_dict = {}

    nm_figure = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/graphs_report/'
    nm_figure_final = nm_figure + 'all_losses' + 'fold_' + str(fold) + '_epoch_' + str(epoch)

    detection_monitoring_plot_all_losses(fig2.ax1, metrics, color_palette, epoch, 0, separate_values_dict, do_validation)
    fig2.savefig(nm_figure_final)


######################################################################################################################
# Create Precision - Recall curve
######################################################################################################################

# Load precision and recall variables
if precision_recall:

    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/experiments/my_dataset/'
    datafile = os.path.join(pth, 'precision.npy')
    precision = np.load(datafile)
    datafile = os.path.join(pth, 'recall.npy')
    recall = np.load(datafile)
    recall = recall[0]
    precision = precision[0]
    fig3 = plt.figure(figsize=(10, 10))
    fig3.ax1 = plt.subplot(111)
    fig3.ax1.set_xlabel('Recall')
    fig3.ax1.set_ylabel('Precision')
    fig3.ax1.set_xlim(0, 1)
    fig3.ax1.grid()
    fig3.ax1.set_ylim(0.5, 1)
    color_palette = np.divide(np.array([237, 47, 47]),255)
    color_palette = np.divide(np.array([0, 29, 103]),255)
    fig3.ax1.plot(recall, precision, color=color_palette)
    color_palette = np.divide(np.array([43, 214, 200]), 255)
    color_palette = np.divide(np.array([184, 133, 31]), 255)
    fig3.ax1.scatter(recall, precision, color=color_palette, marker='x')
    plt.show()

    nm_figure = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/graphs_report/'
    nm_figure_final = nm_figure + 'precision_recall_curve3'

    fig3.savefig(nm_figure_final)


######################################################################################################################
# Create F1 - score curve vs. threshold
######################################################################################################################

# Load precision and recall variables and calculate F1 score from it
if f1_score:

    pth = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/medicaldetectiontoolkit/experiments/my_dataset/'
    datafile = os.path.join(pth, 'precision.npy')
    precision = np.load(datafile)
    datafile = os.path.join(pth, 'recall.npy')
    recall = np.load(datafile)
    recall = recall[0]
    precision = precision[0]
    print(recall)
    print(precision)
    F1 = 2 * precision * recall / (precision + recall)
    fig3 = plt.figure(figsize=(10, 10))
    color_palette = np.divide(np.array([237, 47, 47]), 255)
    color_palette = np.divide(np.array([0, 29, 103]), 255)
    print(color_palette)
    color_palette = np.divide(np.array([43, 214, 200]), 255)

    # Figure F score
    thr_vals = np.arange(0.5, 1, 0.01)
    # print("thr", thr_vals[22])
    # print("F1 max:",  F1[22])
    # print("Recall", recall[22])
    # print("Precision:", precision[22])

    fig3.ax1 = plt.subplot(111)
    fig3.ax1.grid()
    fig3.ax1.plot(thr_vals, F1)
    fig3.ax1.set_xlabel('thr')
    fig3.ax1.set_ylabel('F1')

    nm_figure = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/graphs_report/'
    nm_figure_final = nm_figure + 'F1_thr'
    fig3.savefig(nm_figure_final)
#######################################################################################################################
# Create graph to justify threshold for probability map
#######################################################################################################################
if probability_map:
    thr = np.arange(0, 1, 0.1)
    dice_val_all = [0.521546551994927, 0.5296535542700785, 0.5381293703876955, 0.5419669927375307, 0.5435619583355278, 0.5399178686156246, 0.5383030693268567, 0.5293089883623158, 0.5184050986957899, 0.49128379859776944]
    dice_val_tp = [0.7882050455904737, 0.7993485135592776, 0.8115358609840475, 0.8176496236430743, 0.8171509882158795, 0.8066825042446543, 0.8019614804878236, 0.7674840760183452, 0.694555150486255,  0.5110957603933977]
    fig4 = plt.figure(figsize=(10, 10))
    fig4.ax1 = plt.subplot(111)
    fig4.ax1.set_xlabel('Threshold for probability map')
    fig4.ax1.set_ylabel('DICE coefficient')
    fig4.ax1.set_xlim(0, 1)
    fig4.ax1.set_ylim(0.4, 0.9)
    fig4.ax1.grid()
    #fig3.ax1.set_ylim(0.5, 1)
    # Colors for continuous lines
    color_palette1 = np.divide(np.array([237, 47, 47]), 255)
    color_palette2 = np.divide(np.array([0, 29, 103]), 255)
    fig4.ax1.plot(thr, dice_val_all, color=color_palette1)
    fig4.ax1.plot(thr, dice_val_tp, color=color_palette2)
    # Colors scatter
    color_palette3 = np.divide(np.array([43, 214, 200]), 255)
    color_palette4 = np.divide(np.array([184, 133, 31]), 255)
    fig4.ax1.scatter(thr, dice_val_all, color=color_palette4, marker='x')
    fig4.ax1.scatter(thr, dice_val_tp, color=color_palette3, marker='x')

    nm_figure = r'/scratch_net/biwidl311/Cristina_Almagro/pytorch3D/plots/graphs_report/'
    nm_figure_final = nm_figure + 'thr_probability_map2'
    fig4.savefig(nm_figure_final)
