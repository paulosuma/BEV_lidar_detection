import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Auxiliaries.utils import *
import kitti as kt
from Models.pixor_net import PixorNet
from Losses.pixor_loss import Pixor_loss
from multiprocessing import Pool, cpu_count
import time
from Auxiliaries.evaluation import *
from torch.utils.tensorboard import SummaryWriter
tensorwriter = SummaryWriter("./LogsTensor/")
# tensorwriter.add_image("sample image", lidar_input[:,0,:,:])
# tensorwriter.close()



# print(os.cpu_count())

# def compute(x):
#     return x * x

# if __name__ == '__main__':
#     numbers = list(range(1, 1000000))
#     start_time = time.time()

#     with Pool(processes=6) as pool:  # Create a pool with 4 processes
#         results = pool.map(compute, numbers)
#     print(type(results))
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")

 



Kitti_train = kt.Kitti(train=True)

######### Test 1 sample #############
idx = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = PixorNet().to(device)
criterion = Pixor_loss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=1)

load_checkpoint(net, optimizer, scheduler, "./Checkpoints/final_best_detector.pth.tar")

# for idx in range(20):
traindata = (torch.tensor(data).to(device) for data in Kitti_train[idx])
lidar_input, label, gt_boxes = traindata
lidar_input, label = lidar_input.unsqueeze(0).permute(0, 3, 1, 2), label.unsqueeze(0)
pred = net(lidar_input).permute(0, 2, 3, 1)
total_loss, cls_loss, reg_loss = criterion(pred, label)
print(total_loss.item(), cls_loss.item(), reg_loss.item())

pred = pred.cpu().detach().numpy()
gt_boxes = gt_boxes.cpu().detach().numpy()
label = label.cpu().detach().numpy()[0]

bboxes, valid_pred, Orient = filter_predictions(pred.squeeze(0), DETECTION_THRESH)
winning_boxes, winning_orient = perform_nms(bboxes, valid_pred, Orient, NMS_THRESH)
raw_lidar = Kitti_train.get_raw_lidardata_only(idx)
# draw_Box3D_orientation3D_on_LidarBEV(raw_lidar, winning_boxes, winning_orient, gt_boxes)
plot_gt_and_final_boxes(winning_boxes, winning_orient, gt_boxes)
# print("AP ", compute_ap(winning_boxes, gt_boxes))

# ########## Test Batch ###########
# kitti_train_loader = DataLoader(Kitti_train,batch_size=5,shuffle=False, collate_fn=Kitti_train.collate_fn)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net = PixorNet().to(device)
# criterion = Pixor_loss()
# optimizer = torch.optim.Adam(net.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=1)

# load_checkpoint(net, optimizer, scheduler, "./Checkpoints/final_best_detector.pth.tar")

# traindata = (data.to(device) for data in next(iter(kitti_train_loader)))
# lidar_input, _, gt_boxes = traindata
# gt_boxes = gt_boxes.cpu().detach().numpy()
# lidar_input = lidar_input.permute(0, 3, 1, 2) # (N, 36, 800, 700)
# predictions = net(lidar_input)
# predictions = predictions.permute(0, 2, 3, 1) # (N, 200, 175, 7)
# predictions = predictions.cpu().detach().numpy()

# with Pool(processes=cpu_count()) as pool:
#     bboxes_valid_pred = pool.starmap(filter_predictions,[(pred,DETECTION_THRESH) for pred in predictions])
#     winning_boxes = pool.starmap(perform_nms, [(bbxs, v_pred, orient, NMS_THRESH) for bbxs, v_pred, orient in bboxes_valid_pred])
#     aps = pool.starmap(compute_ap, [(wbxes[0], gt_boxes[i]) for i, wbxes in enumerate(winning_boxes)])
#     batch_ap = [ap for ap in aps if ap is not None]

# print(f"batch ap = {np.array(batch_ap).mean()}")





# from evaluation import *



# from mpl_toolkits.mplot3d import Axes3D

# # Define the full voxel grid dimensions
# shape = (20, 17, 7)
# voxels = np.zeros(shape, dtype=bool)

# # Set only the first layer to ones
# voxels[:, :, 0:2] = 1
# voxels[:, :, 3] = 1
# voxels[:, :, 6:] = 1

# # Plotting the voxel grid
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot voxels
# ax.voxels(voxels, edgecolor='k')

# # Set the limits of the plot
# ax.set_xlim([0, 20])
# ax.set_ylim([0, 17])
# ax.set_zlim([0, 7])

# # Hide the axis ticks and labels
# ax.set_xticks([])
# ax.set_yticks([])
# # ax.set_zticks([])

# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()
