import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from Auxiliaries.utils import *
from config import *
from Auxiliaries.pixor_logger import logger

class Kitti(Dataset):
    """ Interface for loading KittiDataset"""

    def __init__(self, train = True, verbose=True):
        self.train = train
        if self.train :
            self.ds_config = Trainingdatapaths()
        else:
            self.ds_config = Testingdatapaths()

        # Load file paths instead of the data
        self.lidar_files = sorted(glob.glob(os.path.join(self.ds_config.LIDAR_DATA, '*.bin')))
        self.image_files = sorted(glob.glob(os.path.join(self.ds_config.IMAGE_DATA, '*.png')))
        self.calib_files = sorted(glob.glob(os.path.join(self.ds_config.CALIB_DATA, '*.txt')))
        if self.train:
            self.label_files = sorted(glob.glob(os.path.join(self.ds_config.LABEL_DATA, '*.txt')))
        self.print_config_data() if verbose else None

    def print_config_data(self):
        config_data = {
            "x_min": X_MIN,
            "x_max": X_MAX,
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "z_min": Z_MIN,
            "z_max": Z_MAX,
            "dL": DL,
            "dW": DW,
            "dH": DH,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
        }
        logger.info("Configuration Settings:")
        for k, v in config_data.items():
            logger.info(f"{k} : {v}")
        logger.info("Kitti initialized!")

    def __len__(self):
        return len(self.lidar_files)
    
    def __getitem__(self, index):
        lidar_data = self.load_lidar_data(self.lidar_files[index]) # an (N, 800, 700, 36)
        # image_data = self.load_image_data(self.image_files[index]) # an (N, 375, 1240, 3)
        calib_data = self.load_calib_data(self.calib_files[index]) # an (N, )
        if self.train:
            label_data, gt_boxes = self.load_label_data(calib_data, self.label_files[index])
            label_data[...,1:] = (label_data[...,1:]-np.array(REG_MEAN))/np.array(REG_STD)
            return lidar_data, label_data, gt_boxes
        else:
            return lidar_data
    
    def load_lidar_data(self, file_path):
        raw_lidar = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        filtered_lidar = filter_lidarpcl(raw_lidar)
        lidar_3Doccupancy_grid = create_3Doccupancy_grid(filtered_lidar)
        lidar_BEV_image = create_lidarBEV_image(filtered_lidar)
        return np.concatenate((lidar_BEV_image, lidar_3Doccupancy_grid), axis=-1)


    def load_image_data(self, file_path):
        im_dimensions = (IMAGE_WIDTH, IMAGE_HEIGHT)
        image = cv2.resize(cv2.imread(file_path), im_dimensions, interpolation=cv2.INTER_LINEAR)
        return image

    def load_calib_data(self,calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        P = lines[2].strip().split()
        P = np.array([float(x) for x in P[1:]]).reshape(3,-1)
        Ro_rect = lines[4].strip().split()
        Ro_rect = np.array([float(x) for x in Ro_rect[1:]]).reshape(3,-1)
        velo_to_cam = lines[5].strip().split()
        velo_to_cam = np.array([float(x) for x in velo_to_cam[1:]]).reshape(3,-1)
        return {'P': P, 'velo_to_cam': velo_to_cam, 'Ro_rect': Ro_rect}
    
    def load_label_data(self, calib, file_path):
        arrays = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                label = parts[0]  # Type
                values = [float(x) for x in parts[1:]]  
                arrays.append([label] + values)
        agents = [Agent(x) for x in arrays]
        target_mask_indices = []
        target_mask_values = []
        gt_boxes = []
        for agent in agents:
            if agent.type == 'Car':
                velo_to_cam = calib['velo_to_cam']
                Ro_rect = calib['Ro_rect']
                cornersBEV = Box3D_lidar_BEV(Box3D_cameraview(agent), velo_to_cam, Ro_rect)
                indices, values, gt_box = get_mask_pixels_and_values(cornersBEV, agent)
                gt_boxes.append(gt_box)
                target_mask_indices.append(indices)
                target_mask_values.append(values)

        L = int((X_MAX-X_MIN)//(OUTPUT_SCALE*DL))+1
        W = int((Y_MAX-Y_MIN)//(OUTPUT_SCALE*DW))+1
        target = np.zeros((W, L, 7))
        for i in range(len(target_mask_indices)):
            x_indices, y_indices = target_mask_indices[i]
            target[x_indices, y_indices] = target_mask_values[i]
        return target, np.array(gt_boxes) # (200, 175, 7)
    
    def get_agents_only(self, idx):
        arrays = []
        with open(self.label_files[idx], 'r') as file:
            for line in file:
                parts = line.split()
                label = parts[0]  # Type
                values = [float(x) for x in parts[1:]]  
                if label == 'Car': arrays.append([label] + values) 
        agents = [Agent(x) for x in arrays]
        return agents
    
    def get_raw_lidardata_only(self, idx):
        return np.fromfile(self.lidar_files[idx], dtype=np.float32).reshape(-1, 4)

    def compute_label_stats(self):
        labels = []
        for i in range(len(self.label_files)):
            calib_data = self.load_calib_data(self.calib_files[i])
            label_data, _ = self.load_label_data(calib_data, self.label_files[i])
            labels.append(label_data)
        labels = np.array(labels)
        print(labels.shape)
        regression_mean = np.mean(labels, axis=(0, 1, 2))
        standard_dev = np.std(labels, axis=(0, 1, 2))
        print(regression_mean)
        print(standard_dev)
    
    def collate_fn(self, data):
        lidar_inputs, labels, gts = zip(*data)

        max_gt_size = max(gt.shape[0] for gt in gts)
        padded_gts = []
        for gt in gts:
            if gt.size == 0:
                padded_gt = np.zeros((max_gt_size, 2, 4))
            else:
                req = max_gt_size - gt.shape[0]
                if req > 0:
                    padded_gt = np.concatenate((gt, np.zeros((req, 2, 4))), axis=0)
                else:
                    padded_gt = gt
            padded_gts.append(padded_gt)
        
        # Convert lists to tensors
        lidar_inputs = torch.tensor(np.stack(lidar_inputs))
        labels = torch.tensor(np.stack(labels))
        padded_gts = torch.tensor(np.stack(padded_gts))
        
        return lidar_inputs, labels, padded_gts





