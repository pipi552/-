from pathlib import Path
import torch
import datetime

class Config:
    def __init__(self, experiment=None, model=None, modal=None, data_path=None, 
                 n_output_joints=24, loss_type=None, mkdir=True,
                 r6d=False, device=None, sensor_input_len=None, test=False):
        
        self.experiment = experiment
        self.model = model
        self.modal = modal
        self.data_path = data_path
        self.root_dir = Path("").absolute()
        self.n_output_joints = 24
        self.test = test
        self.mkdir = mkdir
        self.n_output_joints = n_output_joints
        self.r6d = r6d
        self.sensor_input_len = sensor_input_len

        self.n_input = 0
        self.type = type

        for modal in self.modal:
            self.n_input += self.sensor_input_len[modal]

        if device != None:
            if 'cpu' in device:
                self.device = torch.device(f'cpu')
            if "mps" in device:
                self.device = torch.device('mps')
            else:
                self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.build_paths()

        self.loss_type = loss_type
    
    def build_paths(self):
        self.og_smpl_model_path = self.root_dir / "smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"

        if self.mkdir:
            if self.experiment != None:
                datestring = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
                self.checkpoint_path = self.root_dir / f"checkpoints/{self.experiment}-{datestring}"
                self.checkpoint_path.mkdir(exist_ok=True, parents=True)
            else:
                print("No experiment name give, can't create dir")

    max_sample_len = 300
    acc_scale = 30
    train_pct = 0.9
    val_ratio = 0.1
    batch_size = 128
    torch_seed = 42  # 0
