

## 怎么运行
### 第一步
```
conda create -n ultraposer python=3.10
conda activate ultraposer
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
pip install pytorch_lightning
pip install scipy chumpy wandb pandas openpyxl
```模型下载(https://smpl.is.tue.mpg.de/index.html) and move the .pkl file into the folder .smpl

### 第二步

[Download the ultraposer dataset](https://drive.google.com/file/d/1_TedT0zkkX6Tv5dK-LResU8gDrggYw4d/view?usp=sharing) used for model training.







