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
    R_base = R.from_quat(base_rot_quat)
    R_global = R.from_quat(global_rot_quat)

    pos_local = R_base.inv().apply(global_pos - base_pos)
    R_local = R_base.inv() * R_global

    R_correction = R.from_euler('z', 90, degrees=True)
    R_local_corrected = R_correction * R_local

    rot_local_quat_wxyz = np.roll(R_local_corrected.as_quat(), 1)
    
    # 确保符号一致性
    # 注意: global_rot_quat是xyzw, rot_local_quat_wxyz是wxyz, 需要对齐再做点积
    global_rot_wxyz = np.roll(global_rot_quat, 1, axis=1)
    signs = np.sum(rot_local_quat_wxyz * global_rot_wxyz, axis=1) < 0
    rot_local_quat_wxyz[signs] *= -1

    return pos_local, rot_local_quat_wxyz

def extract_joint_frame(fk_return, name, root_name, fk_model):
    joint_idx = fk_model.model_names.index(name)
    root_idx = fk_model.model_names.index(root_name)
    joint_pos = fk_return.global_translation_extend[0, :, joint_idx, :].cpu().detach().numpy()
    joint_rot = fk_return.global_rotation_extend[0, :, joint_idx, :].cpu().detach().numpy()
    root_pos = fk_return.global_translation_extend[0, :, root_idx, :].cpu().detach().numpy()
    root_rot = fk_return.global_rotation_extend[0, :, root_idx, :].cpu().detach().numpy()
    joint_pos_local, joint_rot_local = transform_frame_quat(joint_pos, joint_rot, root_pos, root_rot)
    return joint_pos_local, joint_rot_local

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data: return None
    framerate = float(entry_data["mocap_framerate"])
    root_trans = entry_data["trans"].reshape(-1, 3)
    pose_raw = entry_data["poses"].reshape(entry_data["poses"].shape[0], -1)
    pose_aa = np.concatenate([pose_raw[:, :72], np.zeros((pose_raw.shape[0], 6))], axis=-1)
    betas = entry_data["betas"]; gender = entry_data["gender"]
    return { "pose_aa": pose_aa, "gender": gender, "trans": root_trans, "betas": betas, "fps": framerate, }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/hdd/zen/data/ActBound/AMASS/AMASS_Complete")
    parser.add_argument("--output_name", type=str, default="amass_all.pkl", help="输出的pkl文件名")
    parser.add_argument("--num_motions", type=int, default=None, help="处理的动作数量上限，用于快速测试。")
    args = parser.parse_args()

    device = torch.device("cuda")
    NEW_ROBOT_ROTATION_AXIS = NEW_ROBOT_ROTATION_AXIS.clone().detach().to(device)
    
    new_robot_joint_names = [ 'pelvis','left_hip_yaw_link', 'left_hip_pitch_link','left_hip_roll_link', 'left_knee_link', 'left_ankle_pitch_link','left_ankle_roll_link','right_hip_yaw_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_pitch_link','right_ankle_roll_link','torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_pitch_link', 'left_elbow_roll_link','left_wrist_pitch_link','left_wrist_yaw_link','right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_pitch_link', 'right_elbow_roll_link','right_wrist_pitch_link','right_wrist_yaw_link',]
    new_robot_joint_names_augment = new_robot_joint_names + ["left_hand_link", "right_hand_link"]
    new_robot_joint_pick = ['pelvis', "left_knee_link", "left_ankle_roll_link", 'right_knee_link', 'right_ankle_roll_link', "left_shoulder_roll_link", "left_elbow_pitch_link", "left_hand_link", 'right_shoulder_roll_link', 'right_elbow_pitch_link', 'right_hand_link']
    smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    new_robot_joint_pick_idx = [ new_robot_joint_names_augment.index(j) for j in new_robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral").to(device)
    shape_new, scale = joblib.load("data/new_robot/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    if not key_name_to_pkls: raise ValueError(f"No motion files found in {amass_root}")

    key_map_items = list(key_name_to_pkls.items())
    if args.num_motions is not None:
        print(f"--- 测试模式: 仅处理前 {args.num_motions} 个动作 ---")
        key_map_items = key_map_items[:args.num_motions]

    new_robot_fk = New_Robot_Humanoid_Batch(device = device)
    
    data_dump = {}
    pbar = tqdm(key_map_items)
    for data_key, file_path in pbar:
        amass_data = load_amass_data(file_path)
        if amass_data is None: continue
        skip = int(amass_data['fps']//30)
        if skip == 0: skip = 1
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        if N < 10: continue
        
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)
        
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)
        dof_pos = torch.zeros((1, N, 27, 1)).to(device)
        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)
        
        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_new_robot_new = torch.cat([gt_root_rot[None, :, None], NEW_ROBOT_ROTATION_AXIS * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            fk_return = new_robot_fk.fk_batch(pose_aa_new_robot_new, root_trans_offset[None, ])
            diff = fk_return['global_translation_extend'][:, :, new_robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss = diff.norm(dim = -1).mean()
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
        
        frame = {}
        for side in ["left", "right"]:
            hand_pos, hand_rot = extract_joint_frame(fk_return, f"{side}_hand_link", "pelvis", new_robot_fk)
            elbow_pos, elbow_rot = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", "pelvis", new_robot_fk)
            hand_shoulder_pos, hand_shoulder_rot = extract_joint_frame(fk_return, f"{side}_hand_link", f"{side}_shoulder_pitch_link", new_robot_fk)
            elbow_shoulder_pos, elbow_shoulder_rot = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", f"{side}_shoulder_pitch_link", new_robot_fk)
            frame[f"{side}_hand_root_pos"]=hand_pos; frame[f"{side}_hand_root_rot"]=hand_rot
            frame[f"{side}_elbow_root_pos"]=elbow_pos; frame[f"{side}_elbow_root_rot"]=elbow_rot
            frame[f"{side}_hand_shoulder_pos"]=hand_shoulder_pos; frame[f"{side}_hand_shoulder_rot"]=hand_shoulder_rot
            frame[f"{side}_elbow_shoulder_pos"]=elbow_shoulder_pos; frame[f"{side}_elbow_shoulder_rot"]=elbow_shoulder_rot
        
        # [核心修正] 在 .numpy() 之前添加 .cpu()
        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_new_robot_new.squeeze().cpu().detach().numpy(),
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30,
                "frame": frame
                }
        
    output_filename = args.output_name
    if args.num_motions is not None:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}_test_{args.num_motions}{ext}"

    joblib.dump(data_dump, os.path.join("data/new_robot", output_filename))
    print(f"\n处理完成，已保存 {len(data_dump)} 个动作到 {output_filename}")