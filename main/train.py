import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, sys

sys.path.append(os.getcwd())
from config import read_yaml
from model import build_model
from data import build_dataloader
from log import get_logger
from utils.utils import AverageMeter
from tensorboardX import SummaryWriter
from datetime import datetime
import shutil
from utils.loss import FocalLoss


class Trainer(object):
    def __init__(self) -> None:
        self.cfg = read_yaml()
        self.trainloader = build_dataloader(self.cfg, data_type="train")
        self.testloader = build_dataloader(self.cfg, data_type="test")
        self.device = torch.device(
            "cuda:{}".format(self.cfg.train.gpu_id)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = build_model(self.cfg).to(self.device)
        self.pretrain = "pretrain" in self.cfg.train.keys()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        # self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = FocalLoss(device=self.device)
        self.logger = get_logger("trainer")
        self.epoch = 0
        self.iter = 0
        self.best_acc = 0
        self.best_loss = float("inf")
        self.avg_loss = AverageMeter()
        self.loss_ratio = 0.5
        self.avg_acc = AverageMeter()
        self.counter = 0
        self.patience = 5
        cur_time = "{0:%Y_%m_%d_%H_%M_%S/}".format(datetime.now())
        self.exp_dir = os.path.join(self.cfg.train.log_dir, cur_time)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)

        self.writer = self.create_writer()

        for file in ["data", "config", "model", "main", "utils", "assets", "dataset"]:
            shutil.copytree(file, os.path.join(self.exp_dir, file))

        if self.pretrain:
            self.logger.info("loading pretrain weight")
            self.model.load_state_dict(
                torch.load(self.cfg.train.pretrain, map_location="cpu"), strict=False
            )

    def run(self):
        for i in range(self.cfg.train.epoch):
            self.epoch += 1
            self.run_epoch()
            self.scheduler.step()

            if self.epoch % self.cfg.train.test_epoch == 0:
                acc_list = self.eval_training()
                self.logger.info("Eval ALL ACC:{:.2f} ".format(acc_list[0]))
                for n in range(self.cfg.model.num_class):
                    self.logger.info(
                        "class{} acc{:.2f} recall:{:.2f} ".format(
                            n, acc_list[n + 1][0], acc_list[n + 1][1]
                        )
                    )

                self.writer.add_scalar("acc", acc_list[0], self.epoch)
                save_dir = os.path.join(self.exp_dir, self.cfg.train.save_weight_dir)
                save_path = os.path.join(save_dir, "epoch_{}.pth".format(self.epoch))
                torch.save(self.model.state_dict(), save_path)
                if acc_list[0] > self.best_acc:
                    self.best_acc = acc_list[0]
                    self.counter = 0
                    save_path = os.path.join(save_dir, "best_epoch.pth")
                    torch.save(self.model.state_dict(), save_path)
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.logger.info(
                            "There is no imporvement for {} epoch, Early Stop!".format(
                                self.patience
                            )
                        )
                        break

        shutil.copytree("log", os.path.join(self.exp_dir, "log"))

    def run_epoch(self):
        self.model.train()
        self.model.to(self.device)
        self.avg_loss.reset()
        self.avg_acc.reset()

        # loop over the dataset multiple times
        label_list = []
        predict_list = []
        # self.dataloader_count(self.trainloader, data_type="train")
        for batch, data in enumerate(self.trainloader):
            self.iter += 1
            inputs, labels, paths = data
            # self.dataloader_img_show(inputs, labels, paths)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).to(torch.int64)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            predict_list.append(outputs)
            label_list.append(labels)

            label_list_tensor = torch.cat(label_list, dim=0)

            predict_tensor = torch.cat(predict_list, dim=0)
            acc_list = self.calculate_acc(predict_tensor, label_list_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.avg_loss.update(loss.item())

            if batch % 100 == 0:
                self.writer.add_scalar("iter_loss", loss, self.iter)

            self.logger.info(
                "Traning Epoch:{}/{} Current Batch: {}/{} Total Loss:{:.4f} ACC:{:.2f} lr:{:.8f} ".format(
                    self.epoch,
                    self.cfg.train.epoch,
                    batch,
                    len(self.trainloader),
                    self.avg_loss.avg,
                    acc_list[0],
                    self.optimizer.param_groups[0]["lr"],
                )
            )
            for n in range(self.cfg.model.num_class):
                self.logger.info(
                    "class{} acc{:.2f} recall:{:.2f} ".format(
                        n, acc_list[n + 1][0], acc_list[n + 1][1]
                    )
                )

    def get_badcase(self, predict, target, pair_paths, epoch_id):
        import csv

        with torch.no_grad():
            pred_labels = predict.argmax(dim=1)
            bad_pos = pred_labels != target
            bad_list = bad_pos.tolist()
        bad_index = [id for id, i in enumerate(bad_list) if i == True]

        badcase_infos = [
            [pair_paths[i], target.tolist()[i], pred_labels.tolist()[i]]
            for i in bad_index
        ]

        with open("log/eval_badcase.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows([["-----------epoch {}-----------".format(epoch_id)]])
            writer.writerows([["path,gt,predict"]])
            writer.writerows(badcase_infos)

    def create_optimizer(self):
        if self.cfg.train.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
            )
        elif self.cfg.train.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
            )
        elif self.cfg.train.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
            )
        else:
            raise ValueError("no such optimizer,please change in config/congfig.yml")
        return optimizer

    def create_scheduler(self):
        if self.cfg.train.lr_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.9
            )
        elif self.cfg.train.lr_scheduler == "cosinelr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.train.epoch
            )
        else:
            raise ValueError("no such scheduler,please change in config/congfig.yml")
        return scheduler

    def create_writer(self):
        writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))
        return writer

    def eval_training(self):
        self.model.eval()
        self.logger.info("-" * 128)

        label_list = []
        predict_list = []
        paths_list = []
        with torch.no_grad():
            for batch, data in enumerate(self.testloader):
                inputs, labels, paths = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).to(torch.int64)
                label_list.append(labels)
                outputs = self.model(inputs)
                predict_list.append(outputs)
                paths_list.extend(paths)

        label_list_tensor = torch.cat(label_list, dim=0)

        predict_list_tensor = torch.cat(predict_list, dim=0)
        acc_list = self.calculate_acc(predict_list_tensor, label_list_tensor)
        self.get_badcase(predict_list_tensor, label_list_tensor, paths_list, self.epoch)

        return acc_list

    def calculate_acc(self, predict_tensor, label_tensor):
        res = []
        with torch.no_grad():
            predict_index = predict_tensor.argmax(dim=1)
            true_positive = torch.eq(label_tensor, predict_index)
            acc = true_positive.sum() / (true_positive.shape[0] + 1e-3)
            res.append(acc)
            # 获取每一类的predict和label
            for i in range(self.cfg.model.num_class):
                class_predict_index = predict_index == i
                class_label_tensor = label_tensor == i
                class_tp = class_label_tensor * class_predict_index
                class_acc = class_tp.sum() / (class_predict_index.sum() + 1e-3)
                class_recall = class_tp.sum() / (class_label_tensor.sum() + 1e-3)
                res.append([class_acc, class_recall])
        return res


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
