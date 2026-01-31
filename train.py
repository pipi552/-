import os
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from models.UltraPoserModel import UltraPoserModel
from config import Config
from datasets.utils import get_datamodule


seed_everything(42, workers=True)
os.environ["WANDB_MODE"] = "offline"

#parser = get_parser()
#args = parser.parse_args()
start_time = time.time()
fast_dev_run = False
_experiment = "test"
_data_path = ["E:/new_dataset/"]
modal = ["imu","doppler","cir"]
sensor_lens = {"imu":36, "doppler":300, "cir":200}

config = Config(experiment=f"{_experiment}", sensor_input_len=sensor_lens, modal=modal,
            data_path=_data_path, r6d=True, loss_type="mse", device="0", test=False)

model = UltraPoserModel(config)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path

wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)


trainer.fit(model, datamodule=datamodule)

with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
