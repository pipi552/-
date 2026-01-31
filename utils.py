import argparse
import torch
from config import Config
from datetime import datetime
from pathlib import Path
import numpy as np
import shutil


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo_id', help='the combination to run')
    parser.add_argument('--fast_dev_run', default=False, help='fast dev run', action="store_true")
    parser.add_argument('--resume', default=False, help='resume training from checkpoint', action="store_true")
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--device', help='Device ID', default="0")

    return parser

def convert_subset_pose_to_full(config: Config, subset_pose_batch: torch.Tensor):
    ''' expects a batch of poses '''
    B = subset_pose_batch.shape[0] # batch size
    subset_pose_batch = subset_pose_batch.reshape(B, -1, len(config.pred_joints_set), 3, 3)
    _pose = torch.eye(3).repeat(B, subset_pose_batch.shape[1], 24, 1, 1)
    _pose[:, :, config.pred_joints_set] = subset_pose_batch.reshape(B, -1, len(config.pred_joints_set), 3, 3)
    return _pose

def get_checkpoints(combo_id:str, model_names: list, path_to_checkpoints=Path("../../checkpoints/")):
    # find the latest "best" ckpt
    # path_to_checkpoints = Path("../../checkpoints/")

    checkpoints = [x.name for x in path_to_checkpoints.iterdir() if combo_id in x.name]
    
    best_ckpts = {}

    for model_name in model_names:
        model_checkpoints = [x for x in checkpoints if model_name == x.split("_")[0]]

        # get the latest model_checkpoint
        model_creation_dates = [datetime.strptime(x.split("-", 1)[1], "%m%d%Y-%H%M%S") for x in model_checkpoints]

        latest_model = model_checkpoints[np.argmax(model_creation_dates)]

        # now get the best ckpt
        try:
            with open(path_to_checkpoints / latest_model / "best_model.txt", "r") as f:
                best_model_name = Path(f.readlines()[0].strip()).name
            
            best_ckpts[model_name] = path_to_checkpoints / latest_model / best_model_name
        except:
            print("best_model.txt is missing, using the latest checkpoint")
            ckpts = [(x.name.split("=")[2].split("-")[0], x.name) for x in (path_to_checkpoints / latest_model).iterdir() if "epoch" in x.name]
            best_ckpt = sorted(ckpts, key=lambda x: x[0])[-1][1]
            best_ckpts[model_name] = path_to_checkpoints / latest_model / best_ckpt
    
    return best_ckpts

def save_best_models(best_ckpts:list):
    try:
        shutil.rmtree(Path("../../checkpoints/to_send/"))
    except:
        pass
    for k, v in best_ckpts.items():
        path_to_save = Path(f"../../checkpoints/to_send/{k}")
        path_to_save.mkdir(exist_ok=True, parents=True)
        
        shutil.copyfile(v, path_to_save / v.name)



#general rotation matrices
def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


#calculate rotation matrix to take A vector to B vector
def Get_R(A,B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    #get products
    dotprod = np.sum(uA * uB)
    crossprod = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

    #get new unit vectors
    u = uA
    v = uB - dotprod*uA
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0],
                      [crossprod, dotprod, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    #print(R)
    return R

#Same calculation as above using a different formalism
def Get_R2(A, B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)

    return R


#decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R):

    #decomposes as RzRyRx. Note the order: ZYX <- rotation by x first
    thetaz = np.arctan2(R[1,0], R[0,0])
    thetay = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    thetax = np.arctan2(R[2,1], R[2,2])

    return thetaz, thetay, thetax

def Decompose_R_ZXY(R):

    #decomposes as RzRXRy. Note the order: ZXY <- rotation by y first
    thetaz = np.arctan2(-R[0,1], R[1,1])
    thetay = np.arctan2(-R[2,0], R[2,2])
    thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

    return thetaz, thetay, thetax
