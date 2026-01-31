# UltraPoser: Pushing the Limits of IMU-based Full-Body Pose Estimation with Ultrasound Sensing on Consumer Wearables, ACM UIST'25.
### Code and Dataset for UltraPoser [[Paper]](https://yadongli.com/assets/pdf/UltraPoser.pdf) [[Video]](https://www.youtube.com/watch?v=P93l3j6mSCw)

Authors: [Yadong Li*](https://yadongli.com/), Shuning Wang*, [Yongjian Fu](https://fuyongjian.github.io/), Justin Chen, [Xingyu Chen](https://xingyuchen.me/), [Ju Ren](https://juren1987.github.io/), [Xinyu Zhang](https://xyzhang.ucsd.edu/), [Akshay Gadre](https://people.ece.uw.edu/gadre_akshay/index.html), and [Ke Sun](https://samsonsjarkal.github.io/KeSun/)

Affiliation: UW, CSU, UCSD, THU, UMich
  
<div align=center>
    <img src="https://github.com/leeyadong/UltraPoser/blob/f2cf77644b1e1009a5fb2e66e659ad2e4de4e97c/ultraposer.png" alt="method" width="900" />
</div>

## How to Run
### Step 1. Configuration
```
conda create -n ultraposer python=3.10
conda activate ultraposer
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
pip install pytorch_lightning
pip install scipy chumpy wandb pandas openpyxl
```
Download the smpl model from [here](https://smpl.is.tue.mpg.de/index.html) and move the .pkl file into the folder .smpl

### Step 2. Dataset Download

[Download the ultraposer dataset](https://drive.google.com/file/d/1_TedT0zkkX6Tv5dK-LResU8gDrggYw4d/view?usp=sharing) used for model training.

### Step 3. Model training and testing
Modify the data path in train.py, then
```
python train.py
```

For testing, modify the data path and ckpt path in test.py, then
```
python test.py
```




## Acknowledgments
Special thanks to the incredible [IMUPoser](https://github.com/FIGLAB/IMUPoser) and [TransPose](https://github.com/Xinyu-Yi/TransPose) repositories.

## Citing
If you find this code or dataset useful for your research, please consider citing our paper:
```
@inproceedings{ultraposer,
  author = {Li, Yadong and Wang, Shuning and Fu, Yongjian and Chen, Justin and Chen, Xingyu and Ren, Ju and <br> Zhang, Xinyu and Gadre, Akshay and Sun, Ke},
  title = {UltraPoser: Pushing the Limits of IMU-based Full-Body Pose Estimation with Ultrasound Sensing on Consumer Wearables},
  year = {2025},
  booktitle = {ACM Symposium on User Interface Software and Technology},
}
```
