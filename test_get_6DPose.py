import os
import joblib
import os.path as osp
import numpy as np
from tqdm import tqdm

from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_new_robot import MotionLibNewRobot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from scipy.spatial.transform import Rotation as R

def transform_frame(global_pos, global_rot, base_pos, base_rot):
    """
    将 global frame 下的位置和旋转转换为 base frame（如 torso、shoulder）下的局部坐标系。
    输出旋转为四元数（wxyz顺序）

    参数：
        global_pos: np.ndarray (3,)
        global_rot: np.ndarray (3, 3) 或 (4,) – 四元数 (wxyz) or 旋转矩阵
        base_pos: np.ndarray (3,)
        base_rot: np.ndarray (3, 3) 或 (4,)

    返回：
        pos_local: np.ndarray (3,)
        rot_local_quat: np.ndarray (4,) – 四元数 (wxyz)
    """
    # 解析 base 旋转
    if base_rot.shape == (4,):  
        # # 四元数 wxyz → xyzw
        # quat_xyzw = [base_rot[1], base_rot[2], base_rot[3], base_rot[0]]
        
        # 四元数xyzw
        quat_xyzw = base_rot
        R_base = R.from_quat(quat_xyzw).as_matrix()
    else:
        R_base = base_rot

    # 解析 global 旋转
    if global_rot.shape == (4,):
        # # 四元数 wxyz → xyzw
        # quat_xyzw = [global_rot[1], global_rot[2], global_rot[3], global_rot[0]]
        
        # 四元数xyzw
        quat_xyzw = base_rot
        R_global = R.from_quat(quat_xyzw).as_matrix()
    else:
        R_global = global_rot

    # 坐标与旋转转换，由于坐标系的不同，可能需要调整坐标轴顺序，绕 Z 轴顺时针旋转 90° 
    raw_pos_local = R_base.T @ (global_pos - base_pos)
    pos_local = np.array([-raw_pos_local[1], raw_pos_local[0], raw_pos_local[2]])
    rot_local = R_base.T @ R_global

    # 转成四元数 xyzw → 再变 wxyz
    quat_xyzw = R.from_matrix(rot_local).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return pos_local, quat_wxyz

def extract_elbow_trajectory(
    motion_file: str,
    robot_xml_path: str,
    save_path: str,
    dt: float = 1.0 / 30.0,
    num_motions: int = 20
):
    assert os.path.exists(motion_file), f"❌ Motion file {motion_file} 不存在！"
    print(f"✅ Loading {motion_file}")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    sk_tree = SkeletonTree.from_mjcf(robot_xml_path)
    motion_lib = MotionLibNewRobot(
        motion_file=motion_file,
        device=device,
        masterfoot_conifg=None,
        fix_height=False,
        multi_thread=False,
        mjcf_file=robot_xml_path
    )

    motion_lib.load_motions(
        skeleton_trees=[sk_tree] * num_motions,
        gender_betas=[torch.zeros(17)] * num_motions,
        limb_weights=[np.zeros(10)] * num_motions,
        random_sample=False,
        start_idx=0
    )
    motion_keys = motion_lib.curr_motion_keys
    print(f"🎞 加载 {len(motion_keys)} 个动作序列")

    # 获取关节索引
    import ipdb; ipdb.set_trace()
    joint_names = sk_tree._node_names
    idx = lambda name: joint_names.index(name)
    BASE_IDX = idx("pelvis")
    LEFT_SHOULDER_IDX = idx("left_shoulder_pitch_link")
    RIGHT_SHOULDER_IDX = idx("right_shoulder_pitch_link")
    LEFT_ELBOW_IDX = idx("left_elbow_pitch_link")
    RIGHT_ELBOW_IDX = idx("right_elbow_pitch_link")
    LEFT_HAND_IDX = idx("left_wrist_pitch_link")
    RIGHT_HAND_IDX = idx("right_wrist_pitch_link")

    elbow_trajectory_dict = {}

    for motion_id, motion_key in enumerate(motion_keys):
        motion_len_sec = motion_lib.get_motion_length(motion_id).item()
        num_frames = int(motion_len_sec / dt)
        print(f"\n▶ motion_id: {motion_id}, name: {motion_key}, duration: {motion_len_sec:.2f}s, frames: {num_frames}")

        motion_elbow_traj = []

        for t in range(num_frames):
            motion_time = t * dt
            motion_res = motion_lib.get_motion_state(
                torch.tensor([motion_id], device=device),
                torch.tensor([motion_time], device=device)
            )
            rb_pos = motion_res["rg_pos"][0].cpu().numpy()
            rb_rot = motion_res["rb_rot"][0].cpu().numpy()  # xyzw格式
            import ipdb; ipdb.set_trace()
            # 提取base_link的position和rotation
            base_pos = rb_pos[BASE_IDX]
            base_rot = rb_rot[BASE_IDX]

            # 将左右肘关节和朝向转换到base_link坐标系下
            left_elbow_pos = rb_pos[LEFT_ELBOW_IDX]
            left_elbow_rot = rb_rot[LEFT_ELBOW_IDX]
            right_elbow_pos = rb_pos[RIGHT_ELBOW_IDX]
            right_elbow_rot = rb_rot[RIGHT_ELBOW_IDX]
            left_tcp_pos = rb_pos[LEFT_HAND_IDX]
            right_tcp_pos = rb_pos[RIGHT_HAND_IDX]
            left_tcp_rot = rb_rot[LEFT_HAND_IDX]
            right_tcp_rot = rb_rot[RIGHT_HAND_IDX]

            # === 转换到 root_link 坐标系（如 pelvis） ===
            left_elbow_pos_in_root, left_elbow_rot_in_root = transform_frame(left_elbow_pos, left_elbow_rot, base_pos, base_rot)
            right_elbow_pos_in_root, right_elbow_rot_in_root = transform_frame(right_elbow_pos, right_elbow_rot, base_pos, base_rot)
            left_tcp_pos_in_root, left_tcp_rot_in_root = transform_frame(left_tcp_pos, left_tcp_rot, base_pos, base_rot)
            right_tcp_pos_in_root, right_tcp_rot_in_root = transform_frame(right_tcp_pos, right_tcp_rot, base_pos, base_rot)

            # 提取肩部pose，为了后续的相对位置计算
            left_shoulder_pos = rb_pos[LEFT_SHOULDER_IDX]
            left_shoulder_rot = rb_rot[LEFT_SHOULDER_IDX]
            right_shoulder_pos = rb_pos[RIGHT_SHOULDER_IDX]
            right_shoulder_rot = rb_rot[RIGHT_SHOULDER_IDX]

            left_elbow2shoulder_pos, left_elbow2shoulder_rot = transform_frame(left_elbow_pos, left_elbow_rot, left_shoulder_pos, left_shoulder_rot)
            right_elbow2shoulder_pos, right_elbow2shoulder_rot = transform_frame(right_elbow_pos, right_elbow_rot, right_shoulder_pos, right_shoulder_rot)
            left_tcp2shoulder_pos, left_tcp2shoulder_rot = transform_frame(left_tcp_pos, left_tcp_rot, left_shoulder_pos, left_shoulder_rot)
            right_tcp2shoulder_pos, right_tcp2shoulder_rot = transform_frame(right_tcp_pos, right_tcp_rot, right_shoulder_pos, right_shoulder_rot)
            motion_elbow_traj.append({
                "frame": t,
                "time": motion_time,
                "left_root": {
                    "elbow_pos": left_elbow_pos_in_root,
                    "elbow_rot": left_elbow_rot_in_root,
                    "tcp_pos": left_tcp_pos_in_root,
                    "tcp_rot": left_tcp_rot_in_root
                },
                "right_root": {
                    "elbow_pos": right_elbow_pos_in_root,
                    "elbow_rot": right_elbow_rot_in_root,
                    "tcp_pos": right_tcp_pos_in_root,
                    "tcp_rot": right_tcp_rot_in_root
                },
                "left_shoulder": {
                    "elbow_pos": left_elbow2shoulder_pos,
                    "elbow_rot": left_elbow2shoulder_rot,
                    "tcp_pos": left_tcp2shoulder_pos,
                    "tcp_rot": left_tcp2shoulder_rot
                },
                "right_shoulder": {
                    "elbow_pos": right_elbow2shoulder_pos,
                    "elbow_rot": right_elbow2shoulder_rot,
                    "tcp_pos": right_tcp2shoulder_pos,
                    "tcp_rot": right_tcp2shoulder_rot
                }
            })

        elbow_trajectory_dict[motion_key] = {
            "motion_id": motion_id,
            "frames": motion_elbow_traj
        }

    joblib.dump(elbow_trajectory_dict, save_path)
    print(f"\n💾 成功保存 {len(elbow_trajectory_dict)} 段轨迹到 {save_path}")

def extract_training_samples(pkl_path, save_path):
    data = joblib.load(pkl_path)
    print(f"✅ Loaded {len(data)} trajectories from {pkl_path}")

    all_samples = []

    for motion_key, motion_data in tqdm(data.items(), desc="Processing trajectories"):
        frames = motion_data["frames"]
        num_frames = len(frames)

        for t in range(num_frames):
            # ==== 1. 构造前5帧的 elbow + TCP 数据 ====
            seq_feats = []
            for k in range(4, -1, -1):  # t-4 to t
                idx = max(0, t - k)
                f = frames[idx]
                feat = np.concatenate([
                    f["left_shoulder"]["elbow_pos"],                  # (3,)
                    f["right_shoulder"]["elbow_pos"],                 # (3,)
                    f["left_shoulder"]["tcp_pos"],             # (3,)
                    f["right_shoulder"]["tcp_pos"],            # (3,)
                    f["left_shoulder"]["tcp_rot"].reshape(-1), # (9,)
                    f["right_shoulder"]["tcp_rot"].reshape(-1) # (9,)
                ])  # shape = 30
                seq_feats.append(feat)

            input_seq = np.stack(seq_feats, axis=0)  # shape = (5, 30)

            # ==== 2. 添加 t+1 帧 的 TCP pose ====
            next_idx = min(t + 1, num_frames - 1)
            f_next = frames[next_idx]

            next_tcp = np.concatenate([
                f_next["left_shoulder"]["tcp_pos"],             # (3,)
                f_next["left_shoulder"]["tcp_rot"].reshape(-1), # (9,)
                f_next["right_shoulder"]["tcp_pos"],            # (3,)
                f_next["right_shoulder"]["tcp_rot"].reshape(-1) # (9,)
            ])  # shape = 24

            # ==== 3. 构造输出（t+1的 elbow pose）====
            target_elbow = np.concatenate([
                f_next["left_shoulder"]["elbow_pos"],                                   # (3,)
                f_next["right_shoulder"]["elbow_pos"],                                  # (3,)
            ])  # shape = 6

            sample = {
                "input": {
                    "seq": input_seq.astype(np.float32),     # (5, 30)
                    "next_tcp": next_tcp.astype(np.float32)  # (24,)
                },
                "target": target_elbow.astype(np.float32)     # (6,)
            }
            all_samples.append(sample)

    print(f"\n💾 Saving {len(all_samples)} samples to {save_path}")
    torch.save(all_samples, save_path)

def extract_and_save_training_samples(pkl_path, save_path):
    data = joblib.load(pkl_path)
    print(f"✅ Loaded {len(data)} trajectories from {pkl_path}")

    all_samples = []

    # rot 以四元数xyzw形式存储
    for motion_key, motion_data in tqdm(data.items(), desc="Processing trajectories"):
        frames = motion_data["frame"]
        num_frames = frames["left_hand_shoulder_pos"].shape[0]

        for t in range(num_frames):
            # ==== 1. 构造前5帧的 elbow + TCP 数据 ====
            seq_feats = []
            for k in range(4, -1, -1):  # t-4 to t
                idx = max(0, t - k)
                feat = np.concatenate([
                    frames["left_hand_shoulder_pos"][idx],       # (3,)
                    frames["left_hand_shoulder_rot"][idx],       # (4,)
                    frames["right_hand_shoulder_pos"][idx],      # (3,)
                    frames["right_hand_shoulder_rot"][idx],      # (4,)
                    frames["left_elbow_shoulder_pos"][idx],      # (3,)
                    frames["left_elbow_shoulder_rot"][idx],      # (4,)
                    frames["right_elbow_shoulder_pos"][idx],     # (3,)
                    frames["right_elbow_shoulder_rot"][idx],     # (4,)
                ])  # shape = 28
                seq_feats.append(feat)

            input_seq = np.stack(seq_feats, axis=0)  # shape = (5, 28)

            # ==== 2. 添加 t+1 帧 的 TCP pose ====
            next_idx = min(t + 1, num_frames - 1)

            next_tcp = np.concatenate([
                frames["left_hand_shoulder_pos"][next_idx],     # (3,)
                frames["left_hand_shoulder_rot"][next_idx],     # (4,)
                frames["right_hand_shoulder_pos"][next_idx],    # (3,)
                frames["right_hand_shoulder_rot"][next_idx],    # (4,)
            ])  # shape = 14

            # ==== 3. 构造输出（t+1的 elbow pose）====
            target_elbow = np.concatenate([
                frames["left_elbow_shoulder_pos"][next_idx],    # (3,)
                frames["left_elbow_shoulder_rot"][next_idx],    # (4,)
                frames["right_elbow_shoulder_pos"][next_idx],   # (3,)
                frames["right_elbow_shoulder_rot"][next_idx],   # (4,)
            ])  # shape = 14

            sample = {
                "input": {
                    "seq": input_seq.astype(np.float32),     # (5, 28)
                    "next_tcp": next_tcp.astype(np.float32)  # (14,)
                },
                "target": target_elbow.astype(np.float32)     # (14,)
            }
            all_samples.append(sample)

    print(f"\n💾 Saving {len(all_samples)} samples to {save_path}")
    torch.save(all_samples, save_path)

# ========= 🧩 主函数入口 ========= #
def main():
    # motion_file = "data/new_robot/amass_all.pkl"
    # robot_xml_path = "resources/robots/h1_2/h1_2.xml"
    # hand_elbow_trajectory_store_path = "get_6D_pose_hand_elbow_trajectory_test.pkl"
    
    # # 提取手肘轨迹
    # extract_elbow_trajectory(
    #     motion_file=motion_file,
    #     robot_xml_path=robot_xml_path,
    #     save_path=hand_elbow_trajectory_store_path
    # )
    
    # # 生成训练样本
    # save_path = "hand_elbow_training_data.pt"
    # extract_training_samples(hand_elbow_trajectory_store_path, save_path)

    hand_elbow_trajectory_store_path = 'data/new_robot/amass_all.pkl'
    save_path = "hand_elbow_training_data.pt"
    extract_and_save_training_samples(hand_elbow_trajectory_store_path, save_path)
if __name__ == "__main__":
    main()
