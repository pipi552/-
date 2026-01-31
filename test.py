import torch
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils import get_parser
from config import Config
from datasets.utils import get_datamodule
import pytorch_lightning as pl
from models.UltraPoserModel import UltraPoserModel
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


seed_everything(42, workers=True)  # 42
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

fast_dev_run = False
_experiment = "test"
_data_path = ["E:/new_dataset/"]
ckpt_path = "./checkpoints/test-09212025-181115/"
modal = ["imu","doppler","cir"]
sensor_lens = {"imu":36, "doppler":300, "cir":200}

config = Config(experiment=f"{_experiment}", sensor_input_len=sensor_lens, modal=modal,
            data_path=_data_path, r6d=True, loss_type="mse", device="0", test=True)

ckpt_files = os.listdir(ckpt_path)
min_loss = 100
ckpt_name = ""
for f in ckpt_files:
    if f.find("ckpt")!= -1:
        loss = float(f[f.find("step_loss")+10:-5])
        if loss < min_loss:
            min_loss = loss
            ckpt_name = f
state_dict = torch.load(ckpt_path+ckpt_name)
new_state_dict = {}

for key, value in state_dict["state_dict"].items():
    new_key = key.replace('pretrained_model.', '')
    new_state_dict[new_key] = value

model = UltraPoserModel(config)
model.load_state_dict(new_state_dict)
datamodule = get_datamodule(config)
datamodule.setup()
checkpoint_path = config.checkpoint_path

trainer = pl.Trainer(accelerator="gpu", devices=[0])


trainer.predict(model, dataloaders=datamodule.predict_dataloader())
errors = torch.cat(model.error)
loss_values = torch.tensor([item['loss'].item() for item in model.test_outputs], device='cuda:0')
print(f"Loss: {torch.mean(loss_values):.8f}")
errors = errors.reshape(errors.shape[0]*errors.shape[1], 24)
joint_mean_error = torch.mean(errors, dim=0)*100
print(joint_mean_error)
# Names of the 24 SMPL joints
keypoints = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
    "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
    "Left_Hand", "Right_Hand"
]

overall_mean_error = torch.mean(joint_mean_error).unsqueeze(0)
keypoints.append("Mean_Error")
joint_mean_error = torch.cat((joint_mean_error, overall_mean_error)).cpu().numpy()
data = pd.DataFrame([joint_mean_error], columns=keypoints)
#output_file = "./output/"+_experiment+".xlsx"
#data.to_excel(output_file, index=False)

print("mpjpe(cm):", torch.mean(torch.tensor(model.mpjpe)))
print("mpjre(deg):", torch.mean(torch.tensor(model.mpjre)))
print("mpjve(cm):", torch.mean(torch.tensor(model.mpjve)))


