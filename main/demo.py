import torch
import os, sys

sys.path.append(os.getcwd())
from config import read_yaml
from model import build_model
from data import build_dataloader
import cv2
from PIL import Image
from torchvision.transforms import transforms
import glob
import numpy as np
import dlib


def preprocess(cfg, image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.data.data_mean, std=cfg.data.data_std),
        ]
    )
    return transform(image)


def cv2_convert(image):
    image = cv2.resize(image, (cfg.data.image_size[0], cfg.data.image_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def dinorm(image_tensor, cfg):
    image_t = image_tensor.reshape(112, 112, 3)
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    data = image_t * data_std + data_mean
    data = data * 255
    # data = np.transpose(data, ())
    cv2.imshow("dinorm", data)
    cv2.waitKey(0)


label_dict = {
    0: "nomask",
    1: "mask",
}


if __name__ == "__main__":
    cfg = read_yaml()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "example_images/test_imgs"
    test_images = sorted(glob.glob(os.path.join(root, "*")))
    model = build_model(cfg).to(device)
    model.eval()
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu"), strict=False)
    for image_path in test_images:
        # 读入数据
        image_cv = cv2.imread(image_path)
        image = cv2_convert(image_cv)
        image = Image.fromarray(image)

        # 数据处理
        image_tensor = preprocess(cfg, image).unsqueeze(0).to(device)
        # dinorm(image_tensor, cfg)
        # 传入模型
        pred_output = model(image_tensor)
        pred_conf = torch.softmax(pred_output[0], dim=0)
        pred_id = torch.argsort(pred_output)[0][-1]
        cv2.putText(
            image_cv,
            "{} ".format(label_dict[pred_id.item()]),
            (10, 20),
            1,
            1.0,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            image_cv,
            "{:.2f}".format(pred_conf[pred_id.item()]),
            (10, 50),
            1,
            1.0,
            (0, 255, 0),
            1,
        )
        cv2.imshow("demo", image_cv)
        cv2.waitKey(0)
