import os, sys

sys.path.append(os.getcwd())
from config import read_yaml
from model import build_model
import torch
from onnxsim import simplify
import onnx


def convert_to_onnx(model, cfg):
    h, w = cfg.data.image_size[0], cfg.data.image_size[1]
    dummy_iput = torch.randn(1, 3, h, w)
    input_names = ["input"]
    output_names = ["output"]
    onnx_path = os.path.join(cfg.onnx.save_path, "model.onnx")
    torch.onnx._export(
        model,
        dummy_iput,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=11,
    )
    print("onnx has been saved in %s" % onnx_path)
    # do simplifty
    # use onnxsimplify to reduce reduent model.
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path.replace(".onnx", "_sim.onnx"))


if __name__ == "__main__":
    # 获取模型
    cfg = read_yaml()
    model = build_model(cfg)
    model.eval()
    # 模型加载权重
    model.load_state_dict(
        torch.load(
            "experiments/2024_07_31_11_10_37/assets/best_epoch.pth", map_location="cpu"
        )
    )
    # 模型转onnx
    convert_to_onnx(model, cfg)
