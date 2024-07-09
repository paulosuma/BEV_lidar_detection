from torch.utils.data import Dataset, DataLoader, random_split
from Auxiliaries.utils import *
from Auxiliaries.evaluation import filter_predictions, perform_nms, compute_ap
import kitti as kt
from Models.pixor_net import PixorNet
from Losses.pixor_loss import Pixor_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from multiprocessing import Pool, cpu_count
from torch.utils.tensorboard import SummaryWriter
tensorwriter = SummaryWriter("./LogsTensor/")


class TestPixor():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

    def initialization(self):
        print("INITIALIZATION\n")
        for attr, value in self.__dict__.items():
            print(f'{attr}: {value}')
        
        print("\n")
        self.net = PixorNet().to(self.device)
        checkpoint = torch.load("./Checkpoints/best_detector.pth.tar")
        self.net.load_state_dict(checkpoint["net"])
        self.net.print_no_of_parameters()
        print("Loaded trained model")
        
        print("\n-------DATASET--------")
        Kitti_test = kt.Kitti(train=False)
        print("Total test data size", len(Kitti_test))
        self.kitti_test_loader = DataLoader(Kitti_test, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=self.num_workers, collate_fn=Kitti_test.collate_fn)
        print("loaded %d test lidar data" % len(Kitti_test))

    def evaluate(self):
        print("\nTesting...")
        tic = time.time()
        with torch.no_grad():
            self.net.eval()
            Ap = 0
            for batch_idx, valdata in enumerate(self.kitti_test_loader):
                valdata = (data.to(self.device) for data in valdata)
                lidar_input = valdata
                gt_boxes = gt_boxes.cpu().detach().numpy()
                lidar_input = lidar_input.permute(0, 3, 1, 2) # (N, 36, 800, 700)
                predictions = self.net(lidar_input)
                predictions = predictions.permute(0, 2, 3, 1) # (N, 200, 175, 7)
                predictions = predictions.cpu().detach().numpy()

                with Pool(processes=cpu_count()) as pool:
                    bboxes_valid_pred = pool.starmap(filter_predictions,[(pred,0.5) for pred in predictions])
                    winning_boxes = pool.starmap(perform_nms, [(bbxs, v_pred, 0.3) for bbxs, v_pred in bboxes_valid_pred])
                    aps = pool.starmap(compute_ap, zip(winning_boxes, gt_boxes))
                    batch_ap = [ap for ap in aps if ap is not None]

                Ap += np.array(batch_ap).mean()

                if batch_idx % 20 == 0:
                    print(f"batch ap = {np.array(batch_ap).mean()}")
            Ap = Ap/len(self.kitti_test_loader)
        toc = time.time()
        print(f"Time taken for evaluation: {toc-tic:.3f}")
        return Ap

    def run(self):
        self.initialization()
        Ap = self.evaluate()
        print(f"\nFinal Average Precision = {Ap}")
        print("\nDone Testing!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixor Testing')

    # Hyper Params
    parser.add_argument('-b', '--batch-size', default=20, type=int,metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel', 
                        dest='batch_size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)', dest='num_workers')
    parser.add_argument('--device', default='cuda', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].', dest='device')

    args = parser.parse_args()

    tester = TestPixor(args)
    tester.run()

