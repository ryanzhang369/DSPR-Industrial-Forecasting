import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDynamicGraphConstructor(nn.Module):
    """
    支持通道独立窗口的自适应动态图构建器
    输出窗口大小维度: [B, T, C]
    """

    def __init__(self, input_dim, num_channels, d_model, max_window_size):
        super(AdaptiveDynamicGraphConstructor, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.d_model = d_model
        self.max_window_size = max_window_size

        # 静态节点嵌入（用于基础图结构）
        self.static_node_embed = nn.Parameter(torch.randn(num_channels, d_model))

        # --- 方案二：为融合先验知识和学习图添加一个可学习的门控 ---
        self.static_graph_fusion_gate = nn.Parameter(torch.tensor([0.5]))

        self.window_size_predictor = nn.Sequential(
            nn.Linear(self.input_dim, 32),  # <-- 关键修复：使用传入的 input_dim
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.temporal_mask_generator = nn.Sequential(
            nn.Linear(self.input_dim, max_window_size),  # <-- 关键修复：使用传入的 input_dim
            nn.Softmax(dim=-1)
        )

    def forward(self, multi_scale_features, time_features, physical_adj=None):
        static_adj_matrices, dynamic_adj_matrices = [], [] # <--- 改良1：分离静态和动态邻接矩阵
        adaptive_masks = []
        window_sizes = []

        for scale_idx, features in enumerate(multi_scale_features):
            B, T_i, C, D = features.shape

            # 准备时间特征
            if T_i <= time_features.size(1):
                selected_time_features = time_features[:, :T_i, :]
            else:
                last_time_feat = time_features[:, -1:, :].repeat(1, T_i - time_features.size(1), 1)
                selected_time_features = torch.cat([time_features, last_time_feat], dim=1)

            # === 为每个通道独立预测窗口 ===
            per_channel_window_list = []
            per_channel_mask_list = []

            for c in range(C):
                # 提取第 c 个通道的特征 [B, T_i, D]
                channel_feat = features[:, :, c, :]  # [B, T_i, D]

                # 拼接时间特征
                predictor_input = torch.cat([channel_feat, selected_time_features], dim=-1)

                # --- 健壮性检查：确保拼接后的维度与初始化的 input_dim 一致 ---
                assert predictor_input.shape[-1] == self.input_dim, \
                    f"Dimension mismatch: concatenated input dim ({predictor_input.shape[-1]}) != initialized self.input_dim ({self.input_dim})"

                flat_input = predictor_input.view(B * T_i, -1)

                # 预测窗口大小 [B*T_i, 1] -> [B, T_i]
                offset = self.window_size_predictor(flat_input).squeeze(-1)
                window_c = 1.0 + (self.max_window_size - 1.0) * torch.sigmoid(offset)
                window_c = window_c.view(B, T_i)
                per_channel_window_list.append(window_c)

                # 预测时间掩码 [B*T_i, max_window] -> [B, T_i, max_window]
                mask_c = self.temporal_mask_generator(flat_input).view(B, T_i, self.max_window_size)
                per_channel_mask_list.append(mask_c)

            # 合并所有通道：[C, B, T_i] -> [B, T_i, C]
            window_size = torch.stack(per_channel_window_list, dim=-1)  # [B, T_i, C]
            temporal_mask = torch.stack(per_channel_mask_list, dim=-1)  # [B, T_i, max_window, C]

            # === 改良1：分别构建静态和动态邻接矩阵 ===
            # 1. 构建静态邻接矩阵 (全局唯一，与时间步无关)
            learned_static_adj_raw = torch.matmul(self.static_node_embed, self.static_node_embed.transpose(0, 1))
            learned_static_adj = F.softmax(F.relu(learned_static_adj_raw), dim=1)

            # --- 方案二：如果传入了物理邻接矩阵，则进行融合 ---
            if physical_adj is not None:
                # --- 核心修复：确保物理邻接矩阵与模型在同一设备上 ---
                physical_adj = physical_adj.to(learned_static_adj.device)
                gate = torch.sigmoid(self.static_graph_fusion_gate)
                static_adj = gate * physical_adj + (1 - gate) * learned_static_adj
            else:
                static_adj = learned_static_adj

            # 扩展到与动态图相同的形状 [B, T_i, C, C]
            static_adj_expanded = static_adj.unsqueeze(0).unsqueeze(0).expand(B, T_i, -1, -1)

            # 2. 构建动态邻接矩阵 (随时间步变化)
            dynamic_adj_raw = torch.matmul(features, features.transpose(-2, -1)) / (self.d_model ** 0.5)
            
            # --- 核心修复：在Softmax前屏蔽对角线，强制模型关注非对角线关系 ---
            # 创建一个与邻接矩阵形状相同的对角线掩码
            mask = torch.eye(self.num_channels, self.num_channels, device=features.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, T_i, -1, -1)
            
            # 将对角线元素设置为一个非常小的负数
            dynamic_adj_masked = dynamic_adj_raw.masked_fill(mask, -1e9)
            dynamic_adj = F.softmax(dynamic_adj_masked, dim=-1)
            
            # --- 不再融合，而是分别输出 ---
            static_adj_matrices.append(static_adj_expanded)
            dynamic_adj_matrices.append(dynamic_adj)
            adaptive_masks.append(temporal_mask)
            window_sizes.append(window_size)

        return static_adj_matrices, dynamic_adj_matrices, adaptive_masks, window_sizes