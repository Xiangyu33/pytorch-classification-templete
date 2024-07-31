import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import os, sys

sys.path.append(os.getcwd())
import cv2
from PIL import Image
from config import read_yaml
import pickle as pkl
import numpy as np


class AttriDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data_type):
        super(AttriDataset, self).__init__()
        # 读取pkl
        items = pkl.load(open(cfg.data.data_root + "/%s.pkl" % data_type, "rb"))
        # pkl信息传入list
        self.image_list = []
        self.label_list = []
        self.cfg = cfg
        self.data_type = data_type

        print("{}dataloader len:{}".format(data_type, len(items)))

        for item in items:
            self.image_list.append(item["path"])
            self.label_list.append(item["label"])

        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(45),  # 随机旋转45度
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.data.data_mean, std=cfg.data.data_std),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.data.data_mean, std=cfg.data.data_std),
            ]
        )

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)
        image = self.cv2_convert(image)
        image = Image.fromarray(image)

        if self.data_type == "train":
            image_tensor = self.transform_train(image)
        else:
            image_tensor = self.transform_test(image)

        # 可视化数据
        # unnormalize = transforms.Normalize(
        #     # mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #     std=[1/0.229, 1/0.224, 1/0.225]
        # )
        # to_pil = transforms.ToPILImage()
        # a = unnormalize(image_tensor)
        # a = to_pil(a)
        # a.save("dump/after_trans_{:06d}_{}.png".format(index,self.label_list[index]))

        label_value = self.label_list[index]
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        return image_tensor, label_tensor, image_path

    def __len__(self):
        return len(self.image_list)

    def cv2_resize_convert(self, image):
        image = cv2.resize(
            image, (self.cfg.data.image_size[0], self.cfg.data.image_size[1])
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_normalized_score(self, score_list):
        score_tensor = torch.tensor(score_list, dtype=torch.float32)
        score_min = torch.min(score_tensor)
        score_max = torch.max(score_tensor)
        normalized_score = (score_tensor - score_min) / (score_max - score_min)
        return normalized_score

    def cv2_convert(self, image):
        try:
            image = cv2.resize(
                image, (self.cfg.data.image_size[0], self.cfg.data.image_size[1])
            )
        except Exception as e:
            print(str(e))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
