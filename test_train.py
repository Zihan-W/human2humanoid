import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class HandElbowTrajectoryDataset(Dataset):
    def __init__(self, data_path):
        self.samples = torch.load(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_seq = sample["input"]["seq"]         # (5, 28)
        x_next_tcp = sample["input"]["next_tcp"]  # (14,)
        x_seq = torch.from_numpy(x_seq)
        x_next_tcp = torch.from_numpy(x_next_tcp)

        x_input = torch.cat([x_seq.flatten(), x_next_tcp], dim=0)  # → (5×28 + 14,) = (154,)
        y_target = sample["target"]            # (14,)
        return x_input, y_target

class ElbowMLP(nn.Module):
    def __init__(self, input_dim=154, hidden_dim=256, output_dim=14):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ElbowNNWrapper:
    def __init__(self, input_dim=154, hidden_dim=256, output_dim=14, lr=1e-3, device='cuda'):
        self.model = ElbowMLP(input_dim, hidden_dim, output_dim).to(device)
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset_path, epochs=20, batch_size=128):
        dataset = HandElbowTrajectoryDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x_batch.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"📘 Epoch {epoch:2d} | Loss: {avg_loss:.6f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def predict(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input: shape [B, 154] or [154] tensor
        Returns:
            output: shape [B, 14] or [14]
        """
        self.model.eval()
        with torch.no_grad():
            x_input = x_input.to(self.device)
            if x_input.dim() == 1:
                x_input = x_input.unsqueeze(0)
                output = self.model(x_input)[0]  # 还原单个输出
            else:
                output = self.model(x_input)
        return output

    @staticmethod
    def preprocess_from_pkl(pkl_path, save_path):
        """
        Args:
            pkl_path: Path to the input .pkl file containing motion data.
            save_path: Path to save the processed training data.
        """
        data = joblib.load(pkl_path)
        print(f"✅ Loaded {len(data)} trajectories from {pkl_path}")

        all_samples = []

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

                input_seq = np.stack(seq_feats, axis=0) # shape = (5, 28)

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

        torch.save(all_samples, save_path)
        print(f"\n💾 Saved {len(all_samples)} training samples to {save_path}")

if __name__ == "__main__":
    # 第一步：预处理数据（仅需一次）
    if True:  # 设置为 True 以执行预处理
        hand_elbow_trajectory_store_path = 'data/new_robot/amass_all.pkl'
        preprocessed_save_path = 'hand_elbow_training_data.pt'
        ElbowNNWrapper.preprocess_from_pkl(pkl_path=hand_elbow_trajectory_store_path, save_path=preprocessed_save_path)

    # 第二步：训练模型
    dataset_path = preprocessed_save_path
    model_save_path = "elbow_mlp_model.pth"

    trainer = ElbowNNWrapper()
    trainer.train(dataset_path, epochs=100, batch_size=128)
    trainer.save(model_save_path)
    print("✅ Model training and saving completed.")

    # # 示例：如何加载model并进行预测
    # # 第三步：加载模型并进行预测
    # model = ElbowNNWrapper()
    # model.load("elbow_mlp_model.pth")
    # # 假设你有一个 [154] 输入向量
    # x_input = torch.tensor(my_input_array, dtype=torch.float32)
    # y_pred = model.predict(x_input)

