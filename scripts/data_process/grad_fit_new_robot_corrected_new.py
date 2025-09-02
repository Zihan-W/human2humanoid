# /home/hjj/human2humanoid/scripts/data_process/grad_fit_new_robot_corrected_new.py
# -*- coding: utf-8 -*-
"""
本脚本的核心功能是从AMASS数据集中读取人类动作捕捉数据，
并通过一个基于优化的IK求解过程，将其重定向（Retarget）为适用于H1人形机器动作数据。
最终输出一个.pkl文件，其中包含了机器关节角度序列以及用于后续任务（如神经网络训练）的
预计算好的世界坐标系和局部坐标系下的关节点位姿。

此版本经过修正，包含了对所有关键身体部位（颈部、双肩等）的数据提取，
并加入了断点续传和坏文件容错功能，以保证大规模数据处理的稳定性和效率。

[此为加固版本]：
- 采用了原子化保存策略（写入临时文件后重命名），从根本上防止输出文件在写入过程中损坏。
- 增加了周期性保存功能，减少因意外中断（如断电）导致的数据丢失。
"""

import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import joblib

from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import Rotation as R

# 导入项目相关的模块
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_new_robot_humanoid_batch import New_Robot_Humanoid_Batch, NEW_ROBOT_ROTATION_AXIS
import zipfile # 导入 zipfile 模块以捕获特定错误

# ===================================================================
# ======================== 1. 辅助函数 ============================
# ===================================================================

def transform_frame_quat(global_pos, global_rot_quat_xyzw, base_pos, base_rot_quat_xyzw):
    """
    将一个世界坐标系下的位姿，转换到另一个基座标系下（如肩膀或骨盆），得到其相对位姿。
    
    Args:
        global_pos (np.array): 全局位置 (T, 3)
        global_rot_quat_xyzw (np.array): 全局旋转四元数 (T, 4), 格式为 xyzw
        base_pos (np.array): 基座标系原点的全局位置 (T, 3)
        base_rot_quat_xyzw (np.array): 基座标系的全局旋转四元数 (T, 4), 格式为 xyzw

    Returns:
        tuple: (局部位置 (T, 3), 局部旋转四元数 (T, 4) wxyz格式)
    """
    R_base = R.from_quat(base_rot_quat_xyzw)
    R_global = R.from_quat(global_rot_quat_xyzw)

    pos_local = R_base.inv().apply(global_pos - base_pos)
    R_local = R_base.inv() * R_global

    # 应用一个固定的Z轴旋转修正，这通常是为了匹配不同软件或模型间的坐标系约定
    R_correction = R.from_euler('z', 90, degrees=True)
    R_local_corrected = R_correction * R_local

    # 将scipy的xyzw格式转换为模型常用的wxyz格式
    rot_local_quat_wxyz = np.roll(R_local_corrected.as_quat(), 1, axis=1)
    
    # 确保四元数符号的一致性，避免因双重覆盖性导致的动画跳变
    global_rot_wxyz = np.roll(global_rot_quat_xyzw, 1, axis=1)
    signs = (np.sum(rot_local_quat_wxyz * global_rot_wxyz, axis=1) < 0)
    rot_local_quat_wxyz[signs] *= -1.0

    return pos_local.astype(np.float32), rot_local_quat_wxyz.astype(np.float32)


def extract_joint_frame(fk_return, name, root_name, fk_model):
    """
    从正向动力学(FK)结果中提取一个关节点(name)相对于另一个关节点(root_name)的局部坐标系位姿。
    """
    j_idx = fk_model.model_names.index(name)
    r_idx = fk_model.model_names.index(root_name)
    
    # 从PyTorch张量中提取数据，并转移到CPU上转换为NumPy数组
    joint_pos = fk_return.global_translation_extend[0, :, j_idx, :].detach().cpu().numpy()
    joint_rot_xyzw = fk_return.global_rotation_extend[0, :, j_idx, :].detach().cpu().numpy()
    root_pos = fk_return.global_translation_extend[0, :, r_idx, :].detach().cpu().numpy()
    root_rot_xyzw = fk_return.global_rotation_extend[0, :, r_idx, :].detach().cpu().numpy()
    
    return transform_frame_quat(joint_pos, joint_rot_xyzw, root_pos, root_rot_xyzw)

def load_amass_data(data_path):
    """
    [健壮版] 从AMASS的.npz文件中加载动作数据，能处理损坏的文件。
    """
    try:
        entry = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    except zipfile.BadZipFile:
        # 如果文件是损坏的 .npz 文件，则捕获这个特定的错误
        print(f"\n[警告] 文件损坏，已跳过: {data_path}")
        return None # 返回None，让主循环知道应该跳过这个文件
    except Exception as e:
        # 捕获其他可能的读取错误
        print(f"\n[警告] 读取文件时发生未知错误，已跳过: {data_path} | 错误: {e}")
        return None
        
    if "mocap_framerate" not in entry: return None
    fps = float(entry["mocap_framerate"])
    trans = entry["trans"].reshape(-1, 3)
    poses = entry["poses"].reshape(entry["poses"].shape[0], -1)
    # AMASS的poses包含全身姿态，我们只取SMPL标准的前72个自由度
    pose_aa = np.concatenate([poses[:, :72], np.zeros((poses.shape[0], 6))], axis=-1)
    
    return {"pose_aa": pose_aa, "trans": trans, "betas": entry["betas"], "gender": entry["gender"], "fps": fps}


def synthesize_hand_world_poses_from_yaw(fk_return, model_names, device):
    """
    根据URDF中的定义，从腕部最后一个可动关节(wrist_yaw_link)的位姿，
    通过一个固定的坐标变换，来合成出手部末端执行器(hand_base_link)的精确世界位姿。
    这确保了我们用于优化的手部目标与机器人的真实末端语义完全一致。
    """
    B, N = fk_return.global_translation_extend.shape[0:2]
    
    # 获取左右手腕yaw关节的世界位姿
    idx_L_yaw = model_names.index("left_wrist_yaw_link")
    idx_R_yaw = model_names.index("right_wrist_yaw_link")
    p_yaw_L = fk_return.global_translation_extend[0, :, idx_L_yaw, :]
    q_yaw_L = fk_return.global_rotation_extend[0, :, idx_L_yaw, :] # xyzw格式
    p_yaw_R = fk_return.global_translation_extend[0, :, idx_R_yaw, :]
    q_yaw_R = fk_return.global_rotation_extend[0, :, idx_R_yaw, :]

    # URDF中定义的从wrist_yaw到hand_base的固定平移
    offset = torch.tensor([0.054, 0.0, 0.0], dtype=torch.float32, device=device)[None, :]
    # 将这个局部平移向量旋转到世界坐标系下
    delta_L = quat_apply_xyzw(q_yaw_L, offset)
    delta_R = quat_apply_xyzw(q_yaw_R, offset)
    # 计算出手部的世界坐标
    p_hand_L = p_yaw_L + delta_L
    p_hand_R = p_yaw_R + delta_R

    # URDF中定义的从wrist_yaw到hand_base的固定旋转
    q_fix_L = torch.tensor(R.from_euler('z', 90, degrees=True).as_quat(), dtype=torch.float32, device=device)[None, :]
    q_fix_R = torch.tensor(R.from_euler('xyz', [180, 0, -90], degrees=True).as_quat(), dtype=torch.float32, device=device)[None, :]
    # 将固定旋转应用到手腕旋转上，得到手部的世界旋转
    q_hand_L = quat_mul_xyzw(q_yaw_L, q_fix_L.expand_as(q_yaw_L))
    q_hand_R = quat_mul_xyzw(q_yaw_R, q_fix_R.expand_as(q_yaw_R))

    return (
        p_hand_L.detach().cpu().numpy().astype(np.float32),
        q_hand_L.detach().cpu().numpy().astype(np.float32),
        p_hand_R.detach().cpu().numpy().astype(np.float32),
        q_hand_R.detach().cpu().numpy().astype(np.float32),
    )

# --- PyTorch下的四元数运算工具 ---
def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1); x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2; y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)

def quat_apply_xyzw(q, v):
    qvec = q[..., :3]; qw = q[..., 3:4]
    t = 2.0 * torch.cross(qvec, v.expand_as(qvec), dim=-1)
    return v.expand_as(qvec) + qw * t + torch.cross(qvec, t, dim=-1)


def save_data_atomically(data, path):
    """
    [新增] 使用原子化写入方式安全地保存数据，避免文件损坏。
    先写入临时文件，成功后再重命名为正式文件。
    """
    temp_path = path + ".tmp"
    try:
        joblib.dump(data, temp_path)
        os.rename(temp_path, path)
    except Exception as e:
        print(f"\n[严重错误] 保存到 {path} 失败: {e}")
        # 如果保存失败，尝试删除可能已损坏的临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"已删除损坏的临时文件: {temp_path}")
            except OSError as oe:
                print(f"删除临时文件 {temp_path} 时出错: {oe}")


# ===================================================================
# ======================== 3. 主执行逻辑 ============================
# ===================================================================

if __name__ == "__main__":
    
    # --- 3a. 参数解析 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/home/hjj/human2humanoid/data/AMASS/AMASS_Complete", help="AMASS数据集的根目录")
    parser.add_argument("--output_name", type=str, default="amass_all_corrected_new_full_final.pkl", help="输出的PKL文件名")
    parser.add_argument("--num_motions", type=int, default=None, help="本次运行处理的动作数量上限")
    parser.add_argument("--iters", type=int, default=500, help="每条动作的优化迭代次数")
    parser.add_argument("--save_interval", type=int, default=500, help="[新增] 每处理N个文件就保存一次进度")
    args = parser.parse_args()

    # --- 3b. 初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    NEW_ROBOT_ROTATION_AXIS = NEW_ROBOT_ROTATION_AXIS.clone().detach().to(device)

    new_robot_joint_names = [
        'pelvis',
        'left_hip_yaw_link', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
        'right_hip_yaw_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
        'torso_link',
        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_pitch_link', 'left_elbow_roll_link',
        'left_wrist_pitch_link', 'left_wrist_yaw_link',
        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_pitch_link', 'right_elbow_roll_link',
        'right_wrist_pitch_link', 'right_wrist_yaw_link',
    ]
    new_robot_joint_names_augment = new_robot_joint_names + ["left_hand_link", "right_hand_link"]
    new_robot_joint_pick = ['pelvis', "left_knee_link", "left_ankle_roll_link", 'right_knee_link', 'right_ankle_roll_link', "left_shoulder_roll_link", "left_elbow_pitch_link", "left_hand_link", 'right_shoulder_roll_link', 'right_elbow_pitch_link', 'right_hand_link']
    smpl_joint_pick = ["Pelvis","L_Knee","L_Ankle","R_Knee","R_Ankle","L_Shoulder","L_Elbow","L_Hand","R_Shoulder","R_Elbow","R_Hand"]
    new_robot_joint_pick_idx = [new_robot_joint_names_augment.index(j) for j in new_robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral").to(device)
    shape_new, scale = joblib.load("data/new_robot/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)
    new_robot_fk = New_Robot_Humanoid_Batch(device=device)

    # --- 3c. 搜寻AMASS动作文件并实现断点续传 ---
    all_npzs = glob.glob(f"{args.amass_root}/**/*.npz", recursive=True)
    if not all_npzs: raise ValueError(f"No motion files found in {args.amass_root}")
    split_len = len(args.amass_root.split("/")); 
    key_name_to_pkls = {"0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p for p in all_npzs}
    
    output_path = os.path.join("data/new_robot", args.output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"[信息] 发现已存在的输出文件: {output_path}")
        print("      正在加载已有进度...")
        data_dump = joblib.load(output_path)
        print(f"      加载完成。已处理 {len(data_dump)} 个动作。")
    else:
        print("[信息] 未发现已有进度，将从头开始处理。")
        data_dump = {}

    all_keys = list(key_name_to_pkls.keys())
    completed_keys = set(data_dump.keys())
    remaining_keys = [k for k in all_keys if k not in completed_keys]
    
    print(f"总共发现 {len(all_keys)} 个动作，已完成 {len(completed_keys)} 个，剩余 {len(remaining_keys)} 个待处理。")
    
    key_map_items = [(k, key_name_to_pkls[k]) for k in remaining_keys]

    if args.num_motions is not None:
        print(f"--- 测试模式: 将在剩余动作中最多处理 {args.num_motions} 个 ---")
        key_map_items = key_map_items[:args.num_motions]
    
    # --- 3d. 主循环：逐一处理剩余的动作文件 ---
    pbar = tqdm(key_map_items)
    processed_count = 0 # [修改] 此计数器现在只统计本次运行处理的数量
    try:
        for data_key, file_path in pbar:
            amass = load_amass_data(file_path)
            if amass is None: continue

            skip = int(amass["fps"] // 30);
            if skip == 0: skip = 1
            trans = torch.from_numpy(amass["trans"][::skip]).float().to(device); N = trans.shape[0]
            if N < 10: continue
            
            pose_aa_walk = torch.from_numpy(np.concatenate((amass["pose_aa"][::skip, :66], np.zeros((N, 6))), axis=-1)).float().to(device)
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
            offset = joints[:, 0] - trans; root_trans_offset = trans + offset
            gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)
            dof_pos_new = Variable(torch.zeros((1, N, 27, 1), device=device), requires_grad=True)
            optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)

            for it in range(int(args.iters)):
                verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
                pose_aa_new_robot_new = torch.cat([gt_root_rot[None, :, None], NEW_ROBOT_ROTATION_AXIS * dof_pos_new, torch.zeros((1, N, 2, 3), device=device)], dim=2)
                fk_return = new_robot_fk.fk_batch(pose_aa_new_robot_new, root_trans_offset[None, ])
                p_hand_L_world, _, p_hand_R_world, _ = synthesize_hand_world_poses_from_yaw(fk_return, new_robot_fk.model_names, device)
                robot_pos_pick = fk_return["global_translation_extend"][:, :, new_robot_joint_pick_idx].clone()
                lh_pick_i = new_robot_joint_pick.index("left_hand_link")
                rh_pick_i = new_robot_joint_pick.index("right_hand_link")
                robot_pos_pick[0, :, lh_pick_i, :] = torch.from_numpy(p_hand_L_world).to(robot_pos_pick.device)
                robot_pos_pick[0, :, rh_pick_i, :] = torch.from_numpy(p_hand_R_world).to(robot_pos_pick.device)
                diff = robot_pos_pick - joints[:, smpl_joint_pick_idx]
                loss = diff.norm(dim=-1).mean()
                pbar.set_description_str(f"{it:03d} | Loss(mm): {loss.item()*1000:.2f}")
                optimizer_pose.zero_grad(); loss.backward(); optimizer_pose.step()
                dof_pos_new.data.clamp_(new_robot_fk.joints_range[:, 0, None], new_robot_fk.joints_range[:, 1, None])
            
            # --- 3e. 提取并保存最终结果 ---
            dof_pos_new.data.clamp_(new_robot_fk.joints_range[:, 0, None], new_robot_fk.joints_range[:, 1, None])
            pose_aa_new_robot_new = torch.cat([gt_root_rot[None, :, None], NEW_ROBOT_ROTATION_AXIS * dof_pos_new, torch.zeros((1, N, 2, 3), device=device)], dim=2)
            fk_return = new_robot_fk.fk_batch(pose_aa_new_robot_new, root_trans_offset[None, ])
            root_trans_offset_dump = root_trans_offset.clone()
            root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
            
            p_hand_L_world, q_hand_L_xyzw, p_hand_R_world, q_hand_R_xyzw = synthesize_hand_world_poses_from_yaw(fk_return, new_robot_fk.model_names, device)
            
            frame = {}
            idx_pelvis = new_robot_fk.model_names.index("pelvis"); idx_L_sh = new_robot_fk.model_names.index("left_shoulder_pitch_link"); idx_R_sh = new_robot_fk.model_names.index("right_shoulder_pitch_link")
            pelvis_pos = fk_return.global_translation_extend[0, :, idx_pelvis, :].detach().cpu().numpy().astype(np.float32)
            pelvis_rot_xyzw = fk_return.global_rotation_extend[0, :, idx_pelvis, :].detach().cpu().numpy().astype(np.float32)
            L_sh_pos = fk_return.global_translation_extend[0, :, idx_L_sh, :].detach().cpu().numpy().astype(np.float32)
            L_sh_rot_xyzw = fk_return.global_rotation_extend[0, :, idx_L_sh, :].detach().cpu().numpy().astype(np.float32)
            R_sh_pos = fk_return.global_translation_extend[0, :, idx_R_sh, :].detach().cpu().numpy().astype(np.float32)
            R_sh_rot_xyzw = fk_return.global_rotation_extend[0, :, idx_R_sh, :].detach().cpu().numpy().astype(np.float32)

            for side in ["left", "right"]:
                hand_pos_w = p_hand_L_world if side == "left" else p_hand_R_world
                hand_rot_xyzw = q_hand_L_xyzw if side == "left" else q_hand_R_xyzw
                sh_pos = L_sh_pos if side == "left" else R_sh_pos
                sh_rot_xyzw = L_sh_rot_xyzw if side == "left" else R_sh_rot_xyzw
                
                hand_root_pos, hand_root_rot_wxyz = transform_frame_quat(hand_pos_w, hand_rot_xyzw, pelvis_pos, pelvis_rot_xyzw)
                elbow_root_pos, elbow_root_rot_wxyz = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", "pelvis", new_robot_fk)
                frame[f"{side}_hand_root_pos"] = hand_root_pos; frame[f"{side}_hand_root_rot"] = hand_root_rot_wxyz
                frame[f"{side}_elbow_root_pos"] = elbow_root_pos; frame[f"{side}_elbow_root_rot"] = elbow_root_rot_wxyz
                
                hand_sh_pos, hand_sh_rot_wxyz = transform_frame_quat(hand_pos_w, hand_rot_xyzw, sh_pos, sh_rot_xyzw)
                elbow_sh_pos, elbow_sh_rot_wxyz = extract_joint_frame(fk_return, f"{side}_elbow_pitch_link", f"{side}_shoulder_pitch_link", new_robot_fk)
                frame[f"{side}_hand_shoulder_pos"] = hand_sh_pos; frame[f"{side}_hand_shoulder_rot"] = hand_sh_rot_wxyz
                frame[f"{side}_elbow_shoulder_pos"] = elbow_sh_pos; frame[f"{side}_elbow_shoulder_rot"] = elbow_sh_rot_wxyz

            world_links = {"neck": "torso_link", "left_shoulder": "left_shoulder_pitch_link", "right_shoulder": "right_shoulder_pitch_link", "left_elbow": "left_elbow_pitch_link", "right_elbow": "right_elbow_pitch_link", "left_wrist": "left_wrist_pitch_link", "right_wrist": "right_wrist_pitch_link"}
            for name, link_name in world_links.items():
                idx = new_robot_fk.model_names.index(link_name)
                pos_world = fk_return.global_translation_extend[0, :, idx, :].detach().cpu().numpy().astype(np.float32)
                rot_xyzw = fk_return.global_rotation_extend[0, :, idx, :].detach().cpu().numpy()
                rot_world_wxyz = np.roll(rot_xyzw, 1, axis=1).astype(np.float32)
                frame[f"{name}_root_pos"] = pos_world; frame[f"{name}_root_rot"] = rot_world_wxyz

            frame["left_hand_root_pos"] = p_hand_L_world; frame["left_hand_root_rot"] = np.roll(q_hand_L_xyzw, 1, axis=1).astype(np.float32)
            frame["right_hand_root_pos"] = p_hand_R_world; frame["right_hand_root_rot"] = np.roll(q_hand_R_xyzw, 1, axis=1).astype(np.float32)

            data_dump[data_key] = {"root_trans_offset": root_trans_offset_dump.squeeze().detach().cpu().numpy().astype(np.float32), "pose_aa": pose_aa_new_robot_new.squeeze().detach().cpu().numpy().astype(np.float32), "dof": dof_pos_new.squeeze().detach().cpu().numpy().astype(np.float32), "root_rot": sRot.from_rotvec(gt_root_rot.detach().cpu().numpy()).as_quat().astype(np.float32), "fps": 30, "frame": frame}
            processed_count += 1
            
            # --- [新增] 周期性保存逻辑 ---
            if processed_count > 0 and processed_count % args.save_interval == 0:
                print(f"\n已处理 {processed_count} 个新文件，达到保存点，正在执行中期保存...")
                save_data_atomically(data_dump, output_path)
                print(f"中期保存完成。当前总进度 {len(data_dump)} 已安全写入磁盘。")

    finally:
        # --- [修改] 最终安全保存 ---
        print(f"\n任务结束或被中断。本次运行共处理了 {processed_count} 个新动作。")
        if processed_count > 0:
            print("正在执行最终的安全保存...")
            save_data_atomically(data_dump, output_path)
            print(f"最终保存完成！现在总共有 {len(data_dump)} 个动作保存在 {output_path} 中。")
        else:
            print("\n本次运行没有处理新的动作，无需保存。")