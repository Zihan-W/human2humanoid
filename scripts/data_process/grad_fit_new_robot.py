import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_new_robot_humanoid_batch import New_Robot_Humanoid_Batch,NEW_ROBOT_ROTATION_AXIS
from torch.autograd import Variable
from tqdm import tqdm
import argparse

from scipy.spatial.transform import Rotation as R
def transform_frame_quat(global_pos, global_rot_quat, base_pos, base_rot_quat):
    """
    将 global frame 下的位置和四元数旋转转换为 base frame（如 pelvis）下的局部坐标系，并进行坐标轴对齐修正。

    参数：
        global_pos: np.ndarray (N, 3)
        global_rot_quat: np.ndarray (N, 4)  # 四元数 wxyz
        base_pos: np.ndarray (N, 3)
        base_rot_quat: np.ndarray (N, 4)

    返回：
        pos_local: np.ndarray (N, 3)
        rot_local_quat: np.ndarray (N, 4)  # wxyz，已对齐符号
    """
    # 将 wxyz → xyzw（因为 scipy 使用 xyzw）
    R_base = R.from_quat(base_rot_quat[:, [1, 2, 3, 0]])
    R_global = R.from_quat(global_rot_quat[:, [1, 2, 3, 0]])

    # 坐标系旋转修正矩阵（针对你说的问题）
    R_fix = R.from_matrix(np.array([
        [-1, 0, 0],  # X 轴反向
        [ 0, 0, 1],  # Y → Z
        [ 0, 1, 0],  # Z → Y
    ]))

    # 位置转换：先 global → base，再 base → robot
    pos_local = R_base.inv().apply(global_pos - base_pos)
    pos_local = R_fix.apply(pos_local)  # 轴修正

    # 旋转转换：global → base → robot
    R_local = R_base.inv() * R_global
    R_local = R_fix * R_local  # 修正轴顺序

    # 转换回 wxyz
    rot_local_quat = R_local.as_quat()[:, [3, 0, 1, 2]]

    # 修正四元数符号歧义
    signs = np.sum(rot_local_quat * global_rot_quat, axis=1) < 0
    rot_local_quat[signs] *= -1

    return pos_local, rot_local_quat

def extract_joint_frame(fk_return, name, root_name, fk_model):
    """
    提取关节在指定 root 坐标系下的位置和旋转（四元数）

    参数：
        fk_return: FK 输出结果
        name: string，目标关节名，如 "left_hand_link"
        root_name: string，参考关节坐标系名，如 "pelvis" 或 "left_shoulder_pitch_link"
        fk_model: forward kinematics 模型实例，含 model_names 列表

    返回：
        joint_pos_local: np.ndarray, (N, 3)
        joint_rot_local: np.ndarray, (N, 4)  # 四元数 wxyz
    """
    joint_idx = fk_model.model_names.index(name)
    root_idx = fk_model.model_names.index(root_name)

    joint_pos = fk_return.global_translation_extend[0, :, joint_idx, :].cpu().detach().numpy()
    joint_rot = fk_return.global_rotation_extend[0, :, joint_idx, :].cpu().detach().numpy()
    root_pos = fk_return.global_translation_extend[0, :, root_idx, :].cpu().detach().numpy()
    root_rot = fk_return.global_rotation_extend[0, :, root_idx, :].cpu().detach().numpy()

    joint_pos_local, joint_rot_local = transform_frame_quat(joint_pos, joint_rot, root_pos, root_rot)
    return joint_pos_local, joint_rot_local

# def load_amass_data(data_path):
#     entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

#     if not 'mocap_framerate' in  entry_data:
#         return
#     framerate = entry_data['mocap_framerate']


#     root_trans = entry_data['trans']
#     pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
#     betas = entry_data['betas']
#     gender = entry_data['gender']
#     N = pose_aa.shape[0]
#     return {
#         "pose_aa": pose_aa,
#         "gender": gender,
#         "trans": root_trans,
#         "betas": betas,
#         "fps": framerate
#     }
def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return

    framerate = float(entry_data["mocap_framerate"])
    root_trans = entry_data["trans"].reshape(-1, 3)
    pose_raw = entry_data["poses"].reshape(entry_data["poses"].shape[0], -1)

    # ✅ 使用标准72维SMPL pose（24 joints × 3）
    pose_aa = np.concatenate(
        [pose_raw[:, :72], np.zeros((pose_raw.shape[0], 6))], axis=-1
    )

    betas = entry_data["betas"]
    gender = entry_data["gender"]

    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/hdd/zen/data/ActBound/AMASS/AMASS_Complete")
    args = parser.parse_args()

    device = torch.device("cuda")

    NEW_ROBOT_ROTATION_AXIS = NEW_ROBOT_ROTATION_AXIS.clone().detach().to(device)
    # TODO：修改为新的机器人模型的关节名称
    new_robot_joint_names = [ 'pelvis',
                   'left_hip_yaw_link', 'left_hip_pitch_link','left_hip_roll_link', 'left_knee_link', 'left_ankle_pitch_link','left_ankle_roll_link',
                   'right_hip_yaw_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_pitch_link','right_ankle_roll_link',
                   'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_pitch_link', 'left_elbow_roll_link','left_wrist_pitch_link','left_wrist_yaw_link',
                  'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_pitch_link', 'right_elbow_roll_link','right_wrist_pitch_link','right_wrist_yaw_link',]

    new_robot_joint_names_augment = new_robot_joint_names + ["left_hand_link", "right_hand_link"]
    new_robot_joint_pick = ['pelvis',
                 "left_knee_link", "left_ankle_roll_link",
                 'right_knee_link', 'right_ankle_roll_link',
                 "left_shoulder_roll_link", "left_elbow_pitch_link", "left_hand_link",
                 'right_shoulder_roll_link', 'right_elbow_pitch_link', 'right_hand_link']
    smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    new_robot_joint_pick_idx = [ new_robot_joint_names_augment.index(j) for j in new_robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)


    shape_new, scale = joblib.load("data/new_robot/shape_optimized_v1.pkl")    # TODO: 修改为新的机器人模型的形状优化保存路径
    shape_new = shape_new.to(device)

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}

    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    new_robot_fk = New_Robot_Humanoid_Batch(device = device)
    ########################################################################
    left_hand_idx = new_robot_fk.model_names.index("left_hand_link")
    right_hand_idx = new_robot_fk.model_names.index("right_hand_link")
    left_elbow_idx = new_robot_fk.model_names.index("left_elbow_pitch_link")
    right_elbow_idx = new_robot_fk.model_names.index("right_elbow_pitch_link")
    pelvis_idx = new_robot_fk.model_names.index("pelvis")

    left_shoulder_idx = new_robot_fk.model_names.index("left_shoulder_pitch_link")
    right_shoulder_idx = new_robot_fk.model_names.index("right_shoulder_pitch_link")
    #######################################################################
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        pose_aa_new_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), N, axis = 1)
        pose_aa_new_robot[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_new_robot = torch.from_numpy(pose_aa_new_robot).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        dof_pos = torch.zeros((1, N, 27, 1)).to(device) # TODO: 修改为新的机器人模型的总关节数

        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)

        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_new_robot_new = torch.cat([gt_root_rot[None, :, None], NEW_ROBOT_ROTATION_AXIS * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            fk_return = new_robot_fk.fk_batch(pose_aa_new_robot_new, root_trans_offset[None, ])

            diff = fk_return['global_translation_extend'][:, :, new_robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim = -1).mean()
            loss = loss_g


            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()

            dof_pos_new.data.clamp_(new_robot_fk.joints_range[:, 0, None], new_robot_fk.joints_range[:, 1, None])

        dof_pos_new.data.clamp_(new_robot_fk.joints_range[:, 0, None], new_robot_fk.joints_range[:, 1, None])
        pose_aa_new_robot_new = torch.cat([gt_root_rot[None, :, None], NEW_ROBOT_ROTATION_AXIS * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        fk_return = new_robot_fk.fk_batch(pose_aa_new_robot_new, root_trans_offset[None, ])

        root_trans_offset_dump = root_trans_offset.clone()

        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

        ######################################################
        frame = {}
        for side in ["left", "right"]:
            hand_pos, hand_rot = extract_joint_frame(fk_return, f"{side}_hand_link", "pelvis", new_robot_fk)
            elbow_pos, elbow_rot = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", "pelvis", new_robot_fk)

            hand_shoulder_pos, hand_shoulder_rot = extract_joint_frame(fk_return, f"{side}_hand_link", f"{side}_shoulder_pitch_link", new_robot_fk)
            elbow_shoulder_pos, elbow_shoulder_rot = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", f"{side}_shoulder_pitch_link", new_robot_fk)

            frame[f"{side}_hand_root_pos"] = hand_pos
            frame[f"{side}_hand_root_rot"] = hand_rot
            frame[f"{side}_elbow_root_pos"] = elbow_pos
            frame[f"{side}_elbow_root_rot"] = elbow_rot

            frame[f"{side}_hand_shoulder_pos"] = hand_shoulder_pos
            frame[f"{side}_hand_shoulder_rot"] = hand_shoulder_rot
            frame[f"{side}_elbow_shoulder_pos"] = elbow_shoulder_pos
            frame[f"{side}_elbow_shoulder_rot"] = elbow_shoulder_rot
        #######################################################

        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_new_robot_new.squeeze().cpu().detach().numpy(),
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30,
                "frame": frame
                }

        # print(f"dumping {data_key} for testing, remove the line if you want to process all data")
        # import ipdb; ipdb.set_trace()
        joblib.dump(data_dump, "data/new_robot/test.pkl")   # TODO: 修改为新的机器人模型的测试数据保存路径

    # import ipdb; ipdb.set_trace()
    joblib.dump(data_dump, "data/new_robot/amass_all.pkl")  # TODO: 修改为新的机器人模型的测试数据保存路径