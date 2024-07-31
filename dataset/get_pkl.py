import os
import glob
import pickle as pkl
import random


def gen_train_test_pkl(pkl_file):
    data = pkl.load(open(pkl_file, "rb"))
    label_count = [0, 0]
    res_obj = []

    for item in data:
        path = item["path"]
        label = item["label"]

        if label == 0:
            label_count[0] += 1
            res_obj.append(dict(path=path, label=label, conf=1.0))
        elif label == 1:
            label_count[1] += 1
            res_obj.append(dict(path=path, label=label, conf=1.0))

    print("训练数据分布：", label_count)

    random.shuffle(res_obj)
    test_res = res_obj[: len(res_obj) // 10]
    train_res = res_obj[len(res_obj) // 10 :]

    with open("dataset/test.pkl", "wb") as f:
        pkl.dump(test_res, f)
    with open("dataset/train.pkl", "wb") as f:
        pkl.dump(train_res, f)


def gen_pkl(pkl_name):
    root1 = os.path.join(os.getcwd(), "example_images/nomask_faces")  # 替换为自己的数据集路径
    root2 = os.path.join(os.getcwd(), "example_images/mask_faces")  # 替换为自己的数据集路径

    imgs1 = glob.glob(os.path.join(root1, "*"))
    imgs2 = glob.glob(os.path.join(root2, "*"))
    imgs = imgs1 + imgs2
    res_obj = []

    for img in imgs:
        if "nomask_faces" in img:  # 替换为自己的文件夹名
            label = 0  # 替换为自己的分类label
        elif "mask_faces" in img:
            label = 1

        res_obj.append(dict(path=img, label=label, conf=1.0))
    random.shuffle(res_obj)
    with open(pkl_name, "wb") as f:
        pkl.dump(res_obj, f)


if __name__ == "__main__":
    pkl_name = "dataset/dataset.pkl"
    gen_pkl(pkl_name)
    gen_train_test_pkl(pkl_name)
