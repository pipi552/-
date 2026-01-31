import pytorch_lightning as pl
from .Models import *
from smpl.parametricModel import ParametricModel
from maths.angular import r6d_to_rotation_matrix
from config import Config
import evaluator


class UltraPoserModel(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()

        sensor_input_len = config.sensor_input_len
        n_pose_output = config.n_output_joints * (6 if config.r6d == True else 9)
        self.pose_evaluator = PoseEvaluator()
        self.batch_size = config.batch_size
        self.ultraposer_model = UltraPoserNetwork(n_input=sensor_input_len, n_output=n_pose_output, n_feature_embed=256)

        self.train_outputs = []
        self.validation_outputs = []
        self.test_outputs = []
        self.pred = []
        self.mpjpe = []
        self.mpjre = []
        self.mpjve = []
        self.gt = []
        self.label = []
        self.error = []

        self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = 3e-4
        self.save_hyperparameters()

    def forward(self, sensor_inputs, sensor_lens):

        pred_pose = self.ultraposer_model(sensor_inputs, sensor_lens)
        return pred_pose

    def training_step(self, batch, batch_idx):

        sensor_inputs, target_pose_rot, input_lengths = batch
        pred_pose_rot = self(sensor_inputs, input_lengths)
        joint_rot_loss = self.loss(pred_pose_rot, target_pose_rot)

        pred_pose_rot = r6d_to_rotation_matrix(pred_pose_rot)
        target_pose_rot = r6d_to_rotation_matrix(target_pose_rot)

        pred_joint_pos = self.bodymodel.forward_kinematics(pose=pred_pose_rot.view(-1, 216))[1]
        target_joint_pos = self.bodymodel.forward_kinematics(pose=target_pose_rot.view(-1, 216))[1]

        joint_pos_loss = self.loss(pred_joint_pos, target_joint_pos)

        loss = joint_rot_loss + joint_pos_loss

        self.log(f"training_step_loss", loss.item(), batch_size=self.batch_size)
        self.train_outputs.append({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        sensor_inputs, target_pose_rot, input_lengths = batch
        pred_pose_rot = self(sensor_inputs, input_lengths)
        joint_rot_loss = self.loss(pred_pose_rot, target_pose_rot)

        pred_pose_rot = r6d_to_rotation_matrix(pred_pose_rot)
        target_pose_rot = r6d_to_rotation_matrix(target_pose_rot)

        pred_joint_pos = self.bodymodel.forward_kinematics(pose=pred_pose_rot.view(-1, 216))[1]
        target_joint_pos = self.bodymodel.forward_kinematics(pose=target_pose_rot.view(-1, 216))[1]

        joint_pos_loss = self.loss(pred_joint_pos, target_joint_pos)

        loss = joint_rot_loss + joint_pos_loss
        self.validation_outputs.append({"loss": loss})

        self.log(f"validation_step_loss", loss.item(), batch_size=self.batch_size)

        return {"loss": loss}

    def predict_step(self, batch, batch_idx):

        sensor_inputs, target_pose_rot, input_lengths = batch
        pred_pose_rot = self(sensor_inputs, input_lengths)
        joint_rot_loss = self.loss(pred_pose_rot, target_pose_rot)

        pred_pose_rot = r6d_to_rotation_matrix(pred_pose_rot)
        target_pose_rot = r6d_to_rotation_matrix(target_pose_rot)

        pred_joint_pos = self.bodymodel.forward_kinematics(pose=pred_pose_rot.view(-1, 216))[1]
        target_joint_pos = self.bodymodel.forward_kinematics(pose=target_pose_rot.view(-1, 216))[1]

        joint_pos_loss = self.loss(pred_joint_pos, target_joint_pos)

        loss = joint_rot_loss + joint_pos_loss

        poseEval = PoseEvaluator()
        Errors = poseEval.eval(pred_pose_rot, target_pose_rot)

        self.mpjpe.append(Errors[2, 0])
        self.mpjre.append(Errors[1, 0])
        self.mpjve.append(Errors[3, 0])

        self.error.append(self.eval_sample_error(pred_pose_rot, target_pose_rot))

        self.test_outputs.append({"loss": loss})

        return {"loss": loss}

    def eval_sample_error(self, pred_pose, target_pose):

        body_model = ParametricModel("smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl", device=torch.device('cuda'))
        p = pred_pose.reshape(-1, 24, 3, 3)
        t = target_pose.reshape(-1, 24, 3, 3)

        self.pred.append(p.cpu())
        prot, pjoint = body_model.forward_kinematics(p)
        trot, tjoint = body_model.forward_kinematics(t)

        pjoint = pjoint.reshape(-1, 10, 24, 3)
        tjoint = tjoint.reshape(-1, 10, 24, 3)

        error = (tjoint - pjoint).norm(dim=3)

        return error

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.train_outputs, loop_type="train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_outputs, loop_type="val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        self.epoch_end_callback(self.test_outputs, loop_type="test")
        self.test_outputs.clear()

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = [output["loss"] for output in outputs]
        # Aggregate the losses
        avg_loss = torch.mean(torch.tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = evaluator.FullMotionEvaluator("./smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
                                                      joint_mask=[18, 19, 20, 21])
        self._eval_fn_batch = evaluator.FullMotionEvaluatorBatch("./smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
                                                                 joint_mask=[18, 19, 20, 21])
        self.joint_err = evaluator.PerJointErrorEvaluator("./smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        # pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        # pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[7], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100, errs[5] / 100])

    def eval_batch(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        errs = self._eval_fn_batch(pose_p, pose_t)
        return errs

    def eval_perjoint(self, joint_p, joint_t):
        return self.joint_err(joint_p, joint_t)

    @staticmethod
    def print(errors, batch_idx):
        torch.save(errors, "./predictions/" + str(batch_idx) + '.pt')
        for i, name in enumerate(['Maksed Joint Error (cm)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Pred (100m/s^3)', 'Jitter Target (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
