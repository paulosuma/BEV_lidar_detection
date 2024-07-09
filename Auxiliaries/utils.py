import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pg
import math
import cv2
from config import *
import torch


class Agent:
    def __init__(self, agent_line):
        self.type       = agent_line[0]  # 'Car', 'Pedestrian', ...
        self.truncation = agent_line[1] # truncated pixel ratio ([0..1])
        self.occlusion  = agent_line[2] # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
        self.alpha      = agent_line[3] # object observation angle ([-pi..pi])

        # extract 2D bounding box in 0-based coordinates
        self.x1 = agent_line[4] # left (pixels)
        self.y1 = agent_line[5] # top
        self.x2 = agent_line[6] # right
        self.y2 = agent_line[7] # bottom

        # 3D object dimensions: height, width, length (in meters)
        self.h    = agent_line[8] # box height, z, lidar
        self.w    = agent_line[9] # box width, y, lidar
        self.l    = agent_line[10] # box length,  x, lidar

        # 3D object location x,y,z in camera coordinates (in meters)
        self.tx = agent_line[11] # location (x)
        self.ty = agent_line[12] # location (y)
        self.tz = agent_line[13] # location (z)

        # Rotation ry around Y-axis in camera coordinates [-pi..pi]
        self.ry   = agent_line[14] # yaw angle

def Box3D_cameraview(agent):
    R = np.array([[math.cos(agent.ry), 0, math.sin(agent.ry), agent.tx],
                   [0, 1,0, agent.ty],
                [-math.sin(agent.ry), 0, math.cos(agent.ry), agent.tz],
                [0, 0, 0, 1]])
    # 3D bounding box dimensions
    l, w, h = agent.l, agent.w, agent.h
    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    ones = np.ones(8)
    vertices = np.array([x_corners, y_corners, z_corners, ones])
    # rotate and translate 3D bounding box
    corners_3D = R @ vertices
    return corners_3D # 4 x 8 matrix
def Orientation3D_cameraview(agent):
    R = np.array([[math.cos(agent.ry), 0, math.sin(agent.ry), agent.tx],
                   [0, 1,0, agent.ty],
                [-math.sin(agent.ry), 0, math.cos(agent.ry), agent.tz],
                [0, 0, 0, 1]])
    # 3D bounding box corners
    x_corners = [0.0, agent.l]
    y_corners = [0.0, 0.0]
    z_corners = [0.0, 0.0]
    ones = np.ones(2)
    vertices = np.array([x_corners, y_corners, z_corners, ones])
    # rotate and translate 3D bounding box
    corners_3D = R @ vertices
    return corners_3D # 4 x 8 matrix
def draw_Box3D_Orientation3D_image(agents, image, Box3D_camview_list, Orientation3D_camview_list, P):
    # PROJECTION TO IMAGE
    def projection(matrix, P):
        corners2D = P @ matrix # 3 x 8
        #scaled projected points
        corners2D[0] /= corners2D[2]
        corners2D[1] /= corners2D[2]
        return corners2D[:2].T
        
    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i, agent in enumerate(agents):
        if np.any(Box3D_camview_list[i][2] < 0.1): # z dimensions behind camera
            return
        projected3Dbox = projection(Box3D_camview_list[i], P)
        projectedOrientation3D = projection(Orientation3D_camview_list[i], P)

        col = OCC_COL[int(agent.occlusion)]
        trc = int(agent.truncation>0.1)

        # Draw the edges on the image
        for edge in EDGES:
            pt1 = projected3Dbox[edge[1]]
            pt2 = projected3Dbox[edge[0]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=col, \
                    linestyle=TRUN_STYLE[trc])
        # Draw the orientation vector
        p1, p2, = projectedOrientation3D[0, 0], projectedOrientation3D[0, 1]
        p3, p4, = projectedOrientation3D[1, 0], projectedOrientation3D[1, 1]
        ax.plot([p1, p3], [p2, p4], 'b-')

    # Set limits and display the plot
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.show()

def draw_Box2D_image(image, agents):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for agent in agents:
        col = OCC_COL[int(agent.occlusion)]
        trc = int(agent.truncation>0.1)
        if agent.type != "Don't care":
            top_left = (agent.x1, agent.y1)
            bottom_right = (agent.x2, agent.y2)

            # Draw the 2D bounding box
            rect = plt.Rectangle(top_left, bottom_right[0] - top_left[0], 
                    bottom_right[1] - top_left[1], linestyle=TRUN_STYLE[trc], 
                    linewidth=2, edgecolor=col, facecolor='none')
            ax.add_patch(rect)
            # # Add a text label
            # label = agent.type
            # ax.text(top_left[0], top_left[1] - 10, label, color=col, fontsize=6, \
            #         verticalalignment='top', bbox=dict(facecolor='yellow', alpha=0.1, 
            #                                            edgecolor='none'))

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.show()

def Box3D_lidar_BEV(Box3D_camview, velo_to_cam, Ro_rect):
    homo = np.array([0, 0, 0, 1])
    T = np.vstack([Ro_rect@velo_to_cam, homo])
    box_lidar_coords = np.linalg.inv(T)@Box3D_camview
    box_lidar_coords = box_lidar_coords[[0,1], :4]
    return box_lidar_coords # 2 x 4 matrix
def Orientation3D_lidar_BEV(Orientation3D_camview, velo_to_cam, Ro_rect):
    homo = np.array([0, 0, 0, 1])
    T = np.vstack([Ro_rect@velo_to_cam, homo])
    box_lidar_coords = np.linalg.inv(T)@Orientation3D_camview
    box_lidar_coords = box_lidar_coords[[0,1], :4]
    return box_lidar_coords # 2 x 4 matrix
def draw_Box3D_orientation3D_on_LidarBEV(raw_lidardata, Box3D_lidarBEV_list, O3D_lidarBEV_list, gt_boxes):
    fig, ax = plt.subplots(figsize=(10, 10))

    for box in Box3D_lidarBEV_list:
        ax.add_patch(pg(box.T, closed=True, edgecolor='r', linewidth=3, facecolor='none', label='Predicted boxes'))
    for box in O3D_lidarBEV_list:
        ax.add_patch(pg(box.T, closed=True, edgecolor='r', linewidth=3, facecolor='none'))
    for box in gt_boxes:
        ax.add_patch(pg(box.T, closed=True, edgecolor='b', linewidth=3, facecolor='none', label='GT boxes'))

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.05), prop={'size': 8})


    x_lidar, y_lidar, intensity = raw_lidardata[:,0], raw_lidardata[:,1], raw_lidardata[:,3]
    condition = x_lidar>0.1
    x_lidar = x_lidar[condition]
    y_lidar = y_lidar[condition]
    intensity = intensity[condition]

    ax.scatter(x_lidar, y_lidar, c=intensity, cmap='viridis', s=1)  # s is the marker size
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Bird's Eye View of LiDAR Point Cloud with Cars")
    plt.axis('equal')
    plt.show()

def get_mask_pixels_and_values(Box3D_lidar_BEV, agent, scale=OUTPUT_SCALE):
    x_lidar = Box3D_lidar_BEV[0]
    x_lidar = np.where(x_lidar>X_MAX,X_MAX, x_lidar)
    x_lidar = np.where(x_lidar<X_MIN,X_MIN, x_lidar)

    y_lidar = Box3D_lidar_BEV[1]
    y_lidar = np.where(y_lidar>Y_MAX, Y_MAX, y_lidar)
    y_lidar = np.where(y_lidar<Y_MIN, Y_MIN, y_lidar)

    gt_box = np.stack((x_lidar, y_lidar), axis=0)

    # image dimensions
    L = int((X_MAX-X_MIN)//(scale*DL))+1
    W = int((Y_MAX-Y_MIN)//(scale*DW))+1
    mask_image = np.zeros((W, L), dtype=np.uint8)

    # lidar coordinates to image coordinates
    x_pixels = ((x_lidar - X_MIN) // (scale*DL)).astype(int)
    y_pixels = W - 1 - (((y_lidar - Y_MIN) // (scale*DW)).astype(int))
    # coordinates = np.array([[y_pixels], [x_pixels]])
    coordinates = np.vstack((x_pixels, y_pixels)).T.reshape((-1, 1, 2))

    # Fill the polygon area with white color
    cv2.fillPoly(mask_image, [coordinates], 255)
    y_idx, x_idx = np.where(mask_image == 255)
    pixel_indices = np.array([y_idx, x_idx])
    theta = agent.ry
    xc, yc = np.mean(x_lidar), np.mean(y_lidar)
    w, l = agent.w, agent.l

    values = []
    for i in range(len(x_idx)):
        dx = xc - ((((x_idx[i]+1)/L)*(X_MAX-X_MIN))+X_MIN) #metric 
        dy = yc - ((((W-1-(y_idx[i]+1))/W)*(Y_MAX-Y_MIN))+Y_MIN)
        data = [1, math.cos(theta), math.sin(theta), dx, dy, math.log(w), math.log(l)]
        values.append(data)

    values = np.array(values)
    return pixel_indices, values, gt_box

def filter_lidarpcl(lidar_pcl):
    x_lidar = lidar_pcl[:,0]
    y_lidar = lidar_pcl[:,1]
    z_lidar = lidar_pcl[:,2]
    mask = (
        (x_lidar >= X_MIN) & (x_lidar <= X_MAX) &
        (y_lidar >= Y_MIN) & (y_lidar <= Y_MAX) &
        (z_lidar >= Z_MIN) & (z_lidar <= Z_MAX)
    )
    return lidar_pcl[mask]

def create_3Doccupancy_grid(filtered_lidar_pcl):
    x_lidar = filtered_lidar_pcl[:,0]
    y_lidar = filtered_lidar_pcl[:,1]
    z_lidar = filtered_lidar_pcl[:,2]

    # grid dimensions
    L = int((X_MAX-X_MIN)//DL)+1
    W = int((Y_MAX-Y_MIN)//DW)+1
    H = int((Z_MAX-Z_MIN)//DH)+1

    occupancy_grid = np.zeros((W, L, H), dtype=np.float32)

    # lidar coordinates to grid coordinates
    x_coords = ((x_lidar - X_MIN) // DL).astype(int)
    y_coords = W - 1 - (((y_lidar - Y_MIN) // DW ).astype(int))
    z_coords = ((z_lidar - Z_MIN) // DH).astype(int)

    occupancy_grid[y_coords, x_coords, z_coords] = 1.0
    
    return occupancy_grid

def create_lidarBEV_image(filtered_lidar_pcl):
    x_lidar = filtered_lidar_pcl[:,0]
    y_lidar = filtered_lidar_pcl[:,1]
    reflectance = filtered_lidar_pcl[:, 3]

    # dimensions
    L = int((X_MAX-X_MIN)//DL)+1
    W = int((Y_MAX-Y_MIN)//DW)+1

    reflectance_image = np.zeros((W, L), dtype=np.float32)
    reflectance_count = np.ones((W, L), dtype=np.float32)

    # lidar coordinates to image coordinates
    x_coords = ((x_lidar - X_MIN) // DL).astype(int)
    y_coords = W - 1 - (((y_lidar - Y_MIN) // DW ).astype(int))

    for i in range(len(x_coords)):
        reflectance_image[y_coords[i], x_coords[i]] += reflectance[i]
        if i == 0:
            pass
        else:
            reflectance_count[y_coords[i], x_coords[i]] += 1

    reflectance_image /= reflectance_count
    return np.expand_dims(reflectance_image, axis=-1)

def visualize_7_featuremaps(target):
    # Plot each slice in the third dimension
    fig, axes = plt.subplots(1, 7, figsize=(21, 3))  # Create a 1x7 grid of subplots
    feature = ['occupancy', 'cos(t)', 'sin(t)', 'dx', 'dy', 'log(w)', 'log(l)']

    for i in range(7):
        ax = axes[i]
        cax = ax.imshow(target[:, :, i], cmap='viridis', aspect='auto')
        ax.set_title(feature[i])
        fig.colorbar(cax, ax=ax)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss






    










            
