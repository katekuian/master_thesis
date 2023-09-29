import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.lines as mlines
import constants
import torch


def get_legend(plant_classification, hex_colors):
    legend_elements = []
    for index, class_name in enumerate(plant_classification):
        element = mlines.Line2D([], [], color=hex_colors[index], marker='s', ls='', label=class_name)
        legend_elements.append(element)

    return legend_elements


def plot_annotation_for_image(im):
    rgb = np.zeros_like(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            index = im[i, j]    
            rgb[i, j] = constants.color_values[index[0]]

    plt.axis('off')
    plt.legend(handles=get_legend(constants.plant_classification, constants.hex_colors), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=9)
    plt.imshow(rgb)
    plt.show()


def plot_ground_truth_annotation(image_name):
    im = cv2.imread(constants.annotation_folder + image_name)
    plot_annotation_for_image(im)


def plot_original_image(image_name):
    im = cv2.imread(constants.image_folder + image_name)
    plt.axis('off')
    plt.imshow(im)


def plot_predicted_segm_mask(pred_seg, image_name, model_type):
    im = pred_seg.detach().cpu().numpy()
    np.unique(im, return_counts=True)
    rgb = np.zeros((*im.shape, 3), dtype=np.uint8)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            index = im[i, j]
            rgb[i, j] = constants.color_values[index]

    plt.axis('off')
    plt.imshow(rgb)
    plt.show()
    plt.savefig(constants.predicted_segm_masks_folder + model_type + '_' + image_name, bbox_inches='tight', pad_inches=0)
