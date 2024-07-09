from torch.utils.data import Dataset, DataLoader, random_split
from Auxiliaries.utils import *
from Auxiliaries.evaluation import filter_predictions, perform_nms, compute_ap, plot_gt_and_final_boxes
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

"""Argparser derived from code by https://github.com/ArashJavan/DeepLIO/blob/master/deeplio/train.py"""

class TrainPixor():
    def __init__(self, args):
        self.learning_rate = args.lr
        self.lr_decay_step = args.lr_decay_step
        self.lr_decay_gamma = args.lr_decay_gamma
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.momentum = args.momentum
        self.beta = args.beta
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.print_freq = args.print_freq
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resume = args.resume
        self.best_detector_path = "./Checkpoints/best_detector.pth.tar"
        self.start_epoch = args.start_epoch

    def initialization(self):
        print("INITIALIZATION\n")
        for attr, value in self.__dict__.items():
            print(f'{attr}: {value}')
        
        print("\n")
        self.net = PixorNet().to(self.device)
        self.net.print_no_of_parameters()

        self.criterion = Pixor_loss(self.gamma, self.beta, self.alpha)
        # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step,
                                                    gamma=self.lr_decay_gamma)
        
        print("\n")
        if self.resume:
            self.start_epoch, self.best_val_loss = load_checkpoint(self.net, self.optimizer, self.scheduler, 
                                            self.best_detector_path)
            steps_taken = self.start_epoch // self.lr_decay_step
            for _ in range(steps_taken):
                self.scheduler.step()
            print(f"Resuming training from epoch {self.start_epoch+1}...")
        else:
            self.best_val_loss = np.inf
        
        print("\n-------DATASET--------")
        self.Kitti_train = kt.Kitti(train=True)

        train_ratio, val_ratio = 0.8, 0.2
        total_size = len(self.Kitti_train)
        print("Total train data size", total_size)
        train_size= int(train_ratio * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(self.Kitti_train, [train_size, val_size])
        self.kitti_train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                        shuffle=True, num_workers=self.num_workers, collate_fn=self.Kitti_train.collate_fn)
        print("Loaded %d training lidar data" % len(train_dataset))

        self.kitti_val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                      shuffle=True, num_workers=self.num_workers, collate_fn=self.Kitti_train.collate_fn)
        print("loaded %d validation lidar data" % len(val_dataset))

    def train(self, epoch):
        tic = time.time()
        self.net.train()
        for batch_idx, traindata in enumerate(self.kitti_train_loader):
            traindata = (data.to(self.device) for data in traindata)
            lidar_input, label, _ = traindata
            lidar_input = lidar_input.permute(0, 3, 1, 2)
            pred = self.net(lidar_input)
            pred = pred.permute(0, 2, 3, 1)
            total_loss, cls_loss, reg_loss = self.criterion(pred, label)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if (batch_idx+1) % self.print_freq == 0:
                console_string = f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.kitti_train_loader)}], "
                console_string += f"Total Loss: {total_loss.item():.3f}, Cls Loss: {cls_loss.item():.3f}, Reg Loss: {reg_loss.item():.3f} "
                console_string += f"Lr: {self.optimizer.param_groups[0]['lr']}"
                print(console_string)
        toc = time.time()
        print(f"Time taken for 1 epoch training: {toc-tic:.3f}")
        # ############    Tensor Board    ############
        tensorwriter = SummaryWriter("./LogsTensor/plots/")
        tensorwriter.add_scalar("Total Loss", total_loss.item(), epoch*len(self.kitti_train_loader)+batch_idx)
        tensorwriter.add_scalar("Cls Loss", cls_loss.item(),epoch*len(self.kitti_train_loader)+batch_idx)
        tensorwriter.add_scalar("Reg Loss", reg_loss.item(),epoch*len(self.kitti_train_loader)+batch_idx)
        tensorwriter.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch*len(self.kitti_train_loader)+batch_idx)
        tensorwriter.close()
    
    def validate(self, epoch):
        tic = time.time()
        with torch.no_grad():
            self.net.eval()
            val_loss = 0.0
            for batch_idx, valdata in enumerate(self.kitti_val_loader):
                valdata = (data.to(self.device) for data in valdata)
                lidar_input, label, _ = valdata
                lidar_input = lidar_input.permute(0, 3, 1, 2)
                pred = self.net(lidar_input)
                pred = pred.permute(0, 2, 3, 1)
                total_loss,_,_ = self.criterion(pred, label)
                val_loss += total_loss.item()
            val_loss /= len(self.kitti_val_loader)
        toc = time.time()
        print(f"Time taken for 1 epoch  validation: {toc-tic:.3f}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"updating best val loss: {self.best_val_loss:.3f}")
            state = {"net" : self.net.state_dict(),"optimizer" : self.optimizer.state_dict(),
                        "scheduler" : self.scheduler.state_dict(),
                    "epoch": epoch + 1,"best_val_loss" : self.best_val_loss}
            torch.save(state, self.best_detector_path)

    def evaluate(self):
        print("\nEvaluating model...")
        tic = time.time()
        with torch.no_grad():
            self.net.eval()
            Ap = 0
            for batch_idx, valdata in enumerate(self.kitti_val_loader):
                valdata = (data.to(self.device) for data in valdata)
                lidar_input, _, gt_boxes = valdata
                gt_boxes = gt_boxes.cpu().detach().numpy()
                lidar_input = lidar_input.permute(0, 3, 1, 2) # (N, 36, 800, 700)
                predictions = self.net(lidar_input)
                predictions = predictions.permute(0, 2, 3, 1) # (N, 200, 175, 7)
                predictions = predictions.cpu().detach().numpy()

                with Pool(processes=cpu_count()) as pool:
                    bboxes_valid_pred = pool.starmap(filter_predictions,[(pred,DETECTION_THRESH) for pred in predictions])
                    winning_boxes = pool.starmap(perform_nms, [(bbxs, v_pred, orient, NMS_THRESH) for bbxs, v_pred, orient in bboxes_valid_pred])
                    aps = pool.starmap(compute_ap, [(wbxes[0], gt_boxes[i]) for i, wbxes in enumerate(winning_boxes)])
                    batch_ap = [ap for ap in aps if ap is not None]

                Ap += np.array(batch_ap).mean()

                if batch_idx % 20 == 0:
                    print(f"batch ap = {np.array(batch_ap).mean()}")
            Ap = Ap/len(self.kitti_val_loader)
        toc = time.time()
        print(f"Time taken for evaluation: {toc-tic:.3f}")
        return Ap
    
    def save_final_model(self, epoch):
        if (epoch+1) % 50 == 0:
            state = {"net" : self.net.state_dict(),"optimizer" : self.optimizer.state_dict(),
                        "scheduler" : self.scheduler.state_dict(),
                    "epoch": epoch + 1,"best_val_loss" : self.best_val_loss}
            torch.save(state, './Checkpoints/detector_epoch%d.pth.tar'%(epoch+1))

    def training(self):
        print("\nTraining...")
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train(epoch)
            self.validate(epoch)
            if (epoch+1) % self.lr_decay_step == 0:
                Ap = self.evaluate()
                self.test_model_on_sample_index(6)
                print(f"Final Average Precision = {Ap}")
            self.save_final_model(epoch)
            self.scheduler.step()

    def test_model_on_sample_index(self, idx=6):
        Kitti_train = kt.Kitti(train=True, verbose=False)

        net = PixorNet().to(self.device)
        checkpoint = torch.load(self.best_detector_path)
        net.load_state_dict(checkpoint["net"])

        traindata = (torch.tensor(data).to(self.device) for data in Kitti_train[idx])
        lidar_input, label, gt_boxes = traindata
        lidar_input, label = lidar_input.unsqueeze(0).permute(0, 3, 1, 2), label.unsqueeze(0)
        pred = net(lidar_input).permute(0, 2, 3, 1)

        pred = pred.cpu().detach().numpy()
        gt_boxes = gt_boxes.cpu().detach().numpy()

        bboxes, valid_pred, Orient = filter_predictions(pred.squeeze(0), DETECTION_THRESH)
        winning_boxes, winning_orient = perform_nms(bboxes, valid_pred, Orient, NMS_THRESH)

        plot_gt_and_final_boxes(winning_boxes, winning_orient, gt_boxes, idx=idx)
        print(f"AP for sample {idx}: {compute_ap(winning_boxes, gt_boxes)}")



    def run(self):
        self.initialization()
        self.training()
        print("Done Training!!")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixor Training')

    # Hyper Params
    parser.add_argument('--lr',default=1e-3,type=float,metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr-decay-step',default=10,type=int,metavar='LR-DECAY-STEP', 
                        help='learning rate decay step',dest='lr_decay_step')
    parser.add_argument('--lr-decay-gamma',default=0.1,type=float,metavar='LR-DECAY-GAMMA', 
                        help='learning rate decay gamma', dest='lr_decay_gamma')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run', dest='epochs')
    parser.add_argument('-b', '--batch-size', default=20, type=int,metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel', 
                        dest='batch_size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)', dest='num_workers')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum', dest='momentum')
    parser.add_argument('--beta', default=1.0, type=float, metavar='B',
                        help='beta for smoothL1_loss', dest='beta')
    parser.add_argument('--gamma', default=3.0, type=float, metavar='G',
                        help='gamma exponential for focal loss', dest='gamma')
    parser.add_argument('--alpha', default=0.99, type=float, metavar='A',
                        help='scaling factore for focal loss', dest='alpha')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency, after N number of batches', 
                        dest='print_freq')
    parser.add_argument('--resume', default=False, type=bool, metavar='R',
                        help='resume training', dest='resume')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='SE',
                        help='the starting epoch', dest='start_epoch')

    args = parser.parse_args()

    trainer = TrainPixor(args)
    trainer.run()

