# pytorch classification templete
This is templete code of pytorch, which used in classification task. 


## Usage
1. pretrained weights download  
you can download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1pRvMG4LvXsaoKHPEm0VulOz5mUTT_fDp?usp=drive_link).Unzip and place them in PROJECT_DIR/weights.

Ensuring the directory is as follows:
```text
classification_pytorch
└── weights
    ├── shufflenetv2_x0.5-f707e7126e.pth
    ├── shufflenetv2_x1-5666bf0f80.pth
    └── mobilenetv3_large_100-427764d5.pth
``` 

2. data prepare  
you can prepare your dataset and modify the path in `dataset/get_pkl.py`

```bash 
python dataset/get_pkl.py

```
the `train.pkl` and `test.pkl` are generated in `dataset` dir

3. train   
set your train config in `config/config.yml`

```bash 
python main/train.py
```

your train weights will be saved in `expriments` dir,named with `time_stamp: YYY_MMMM_DDD_TTT`

the `log` and `eval_badcase` will be saved in `log dir`,you can review the train log.

4. inference  
modify the `demo.py` root and weight path
```bash 
python main/demo.py
```

5. convert onnx
modify the weight path and generate onnx
```bash 
python main/deploy.py 
```
