import torch
import numpy as np
import time
from config import *
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pg
from sklearn.metrics import auc
from multiprocessing import Pool, cpu_count
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def filter_predictions(pred, threshold):
    """pred: (after permutation) (200,175,7) np array"""

    pred[...,1:] = (pred[...,1:]*np.array(REG_STD)) + np.array(REG_MEAN)
    mask = np.where(pred[:,:,0] > threshold)
    chosen_pred = pred[mask]
    if len(mask[0]) == 0:
        print("no qualifying potential detections")
        return np.array([]), np.array([]), np.array([])
    
    y_px, x_px = mask[0], mask[1]
    y_px = 200 - y_px

    x_m = ((x_px/175)*70)
    y_m = ((y_px/200)*80)-40-0.8

    dx = pred[mask][:,3]
    dy = pred[mask][:,4]
    xc, yc = dx + x_m, dy + y_m

    w, l = np.exp(pred[mask][:,6]), np.exp(pred[mask][:,5])
    x_corners = np.stack((l/2, l/2, -l/2, -l/2), axis=1)
    y_corners = np.stack((-w/2, w/2, w/2, -w/2), axis=1)
    ones = np.ones((len(x_corners), 4))
    corners = np.stack((x_corners, y_corners, ones), axis=1)

    ##### Orientation #####
    O_xcorners = np.zeros(x_corners.shape)
    O_ycorners = np.zeros(y_corners.shape)
    O_ycorners[:,2:] = -w[:, np.newaxis]
    O_corners = np.stack((O_xcorners, O_ycorners, ones), axis=1)

    theta = np.arctan2(pred[mask][:,2], pred[mask][:,1]) #atan2(s, c)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    #Rotation plus translation to find corners for each bbox proposal
    bboxes, Orient = np.zeros((len(xc), 2, 4)), np.zeros((len(xc), 2, 4))

    for i in range(len(xc)):
        H = np.array([[cos_t[i], sin_t[i], xc[i]], [-sin_t[i], cos_t[i], yc[i]], [0, 0, 1]])
        bboxes[i] = (H @ corners[i])[:2]
        Orient[i] = (H @ O_corners[i])[:2]

    return bboxes, chosen_pred, Orient # (233, 2, 4)


def boxes_to_polygons(boxes):
    """Convert an array of boxes to an array of Shapely polygons."""
    return [Polygon(bbox.T) for bbox in boxes]

def compute_ious(boxes1, boxes2):
    """boxes1: an Nx2x4 array of boxes each containing 4 corners
       boxes2: an Mx2x4 array of boxes each containing 4 corners"""
    
    # Convert the boxes to polygons
    polygons1 = boxes_to_polygons(boxes1)
    polygons2 = boxes_to_polygons(boxes2)

    N, M = len(polygons1), len(polygons2)
    ious = np.zeros((N, M))

    def compute_iou(polygon1, polygon2):
        """Compute the Intersection over Union (IoU) of two Shapely polygons."""
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area
        return intersection_area / union_area if union_area != 0 else 0

    pairs = [(polygons1[i], polygons2[j]) for i in range(N) for j in range(M)]
    ious = [compute_iou(*pair) for pair in pairs]
    ious = np.array(ious).reshape(N,M)
    # for i in range(N):
    #     for j in range(M):
    #         ious[i, j] = compute_iou(polygons1[i], polygons2[j])

    return ious  # NxM


def perform_nms(bboxes, chosen_pred, Orient, nms_threshold):
    if len(bboxes) == 0:
        return np.array([]), np.array([])
    N_top_boxes = N_TOP_CHOSEN_BOXES
    sorted_indices = np.argsort(chosen_pred[:,0])
    if len(bboxes) > N_top_boxes:
        bboxes_sorted = bboxes[sorted_indices][:N_top_boxes]
        orient_sorted = Orient[sorted_indices][:N_top_boxes]
    else:
        bboxes_sorted = bboxes[sorted_indices]
        orient_sorted = Orient[sorted_indices]

    ious = compute_ious(bboxes_sorted, bboxes_sorted)

    i = 0
    winning_boxes, winning_orient = [], []
    while i < len(bboxes_sorted):
        winning_boxes.append(bboxes_sorted[i])
        winning_orient.append(orient_sorted[i])
        j = i+1
        while j < len(bboxes_sorted):
            if ious[i, j] > nms_threshold:
                bboxes_sorted = np.delete(bboxes_sorted, j, axis=0)
                orient_sorted = np.delete(orient_sorted, j, axis=0)
                ious = np.delete(ious,j, axis=0)
                ious = np.delete(ious,j, axis=1)
            else:
                j+=1
        i+=1
    return np.array(winning_boxes), np.array(winning_orient)

def compute_ap(winning_boxes, gt_boxes):
    N = len(winning_boxes)
    if len(gt_boxes) == 0:
        return
    detections = np.arange(1, N+1)
    precision, recall = [1], [0]
    ious = compute_ious(winning_boxes, gt_boxes)
    positive, negative = [], []

    gt_detected = np.zeros(len(gt_boxes), dtype=bool)

    for i in range(N):
        max_iou_idx = np.argmax(ious[i, :])
        max_iou = ious[i, max_iou_idx]
        if max_iou > 0.1 and not gt_detected[max_iou_idx]:
            positive.append(winning_boxes[i])
            gt_detected[max_iou_idx] = True
        else:
            negative.append(winning_boxes[i])

        precision.append(len(positive)/detections[i])
        recall.append(len(positive)/len(gt_boxes))

    precision, recall = np.array(precision), np.array(recall)

    if len(precision) == 1:
        return

    # Area under the precision-recall curve)
    ap = auc(recall, precision)

    # # Plot the precision-recall curve
    # plt.plot(recall, precision, marker='o')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve (AP = {ap:.2f})')
    # plt.show()
    return ap


def plot_to_tensorboard_image(figure):
    # Convert the plot to a NumPy array
    figure.canvas.draw()
    image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return image



def plot_gt_and_final_boxes(winning_boxes, winning_orient, gt_boxes, window_size=(70, 80), idx=0):
    """
    Plot an array of boxes inside a window.
    
    Parameters:
        boxes (np.ndarray): An array of shape (N x 2 x 4) where each box has 4 corners (x, y).
        window_size (tuple): Size of the larger window (width, height).
    """
    if len(winning_boxes) == 0:
        return
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, window_size[0])
    ax.set_ylim(-window_size[1], window_size[1])
    ax.set_aspect('equal')

    for box in winning_boxes:
        ax.add_patch(pg(box.T, closed=True, edgecolor='r', facecolor='none', label='Predicted boxes'))
    for box in winning_orient:
        ax.add_patch(pg(box.T, closed=True, edgecolor='r', facecolor='none'))
    for box in gt_boxes:
        ax.add_patch(pg(box.T, closed=True, edgecolor='b', facecolor='none', label='GT boxes'))

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.05), prop={'size': 8})

    plt.title('Plot of Predicted and GT Boxes')
    plt.grid(True)

    tensorboard_image = plot_to_tensorboard_image(fig)
    
    # Add image to TensorBoard
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./LogsTensor/boxplots/{curr_time}"
    tensorwriter = SummaryWriter(log_dir)
    tensorwriter.add_image(f'gt&final boxes {idx}', tensorboard_image, global_step=0, dataformats='HWC')
    tensorwriter.close()

    plt.show()
    plt.close(fig)


