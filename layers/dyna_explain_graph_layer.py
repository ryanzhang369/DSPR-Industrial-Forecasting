import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AdaptiveWindowDGNNLayer(nn.Module):
    """
    基于自适应时间窗口的动态图神经网络层
    在动态定义的关系网和时间窗口内，深度挖掘和融合时空信息，生成更强大的特征表示
    """
    def __init__(self, d_model, time_dim, max_window_size, args, num_heads):
        super(AdaptiveWindowDGNNLayer, self).__init__()
        self.d_model = d_model
        self.time_dim = time_dim
        self.max_window_size = max_window_size
        self.num_heads = num_heads  # 别忘了保存这个参数
        # 多头自适应图卷积
        # --- 改良2：为静态图和动态图创建独立的卷积模块 ---
        self.adaptive_graph_convs = nn.ModuleList([
            AdaptiveTemporalGraphConvolution(d_model, d_model // 2, time_dim, max_window_size, args) for _ in range(num_heads)
        ])
        self.static_graph_convs = nn.ModuleList([
            nn.Linear(d_model, d_model // 2) for _ in range(num_heads)])
        # 头融合
        self.head_fusion = nn.Linear(d_model * num_heads, d_model)
        # 残差连接、层归一化和Dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(args.dgn_dr) # 使用args中的dropout率

        # --- 核心修改：添加多尺度注意力融合机制 ---
        # 为每个尺度创建一个可学习的权重
        # +1 是因为包含了原始尺度
        num_scales = 1 + getattr(args, 'down_sampling_layers', 0) # 健壮地获取参数
        self.scale_attention = nn.Parameter(torch.ones(num_scales))

    def forward(self, features_list, static_adj_matrices, dynamic_adj_matrices, temporal_masks, window_sizes, time_encodings, return_gate=False):
        """
        新增参数:
            return_gate (bool): 是否返回门控权重
        Returns:
            output: [B, T, C, D]
            (可选) avg_gate_weights: [B, T, C, D] 或 None
        """
        # 使用原始尺度的时间步数作为基准
        base_time_steps = features_list[0].size(1)  # T
        batch_size, num_channels, d_model = features_list[0].shape[0], features_list[0].shape[2], \
            features_list[0].shape[3]
        multi_scale_outputs = []
        multi_scale_gates = [] if return_gate else None

        # 对每个尺度分别处理
        for scale_idx, (features, static_adj, dynamic_adj, temp_mask, window_size) in enumerate(
                zip(features_list, static_adj_matrices, dynamic_adj_matrices, temporal_masks, window_sizes)
        ):
            scale_time_steps = features.size(1)  # T_i
            head_outputs = []
            head_gates = [] if return_gate else None

            for i in range(self.num_heads):
                dynamic_conv = self.adaptive_graph_convs[i]
                static_conv = self.static_graph_convs[i]

                # --- 核心修改：移除时间步循环，改为批处理 ---
                B, T_i, C, D = features.shape

                features_reshaped = features.view(B * T_i, C, D)  # [B*T_i, C, D]
                dynamic_adj_reshaped = dynamic_adj.view(B * T_i, C, C)  # [B*T_i, C, C]
                static_adj_reshaped = static_adj.view(B * T_i, C, C)
                temporal_mask_reshaped = temp_mask.reshape(B * T_i, C, self.max_window_size)

                if T_i > time_encodings.size(1):
                    last_enc = time_encodings[:, -1:, :].expand(B, T_i - time_encodings.size(1), -1)
                    padded_time_enc = torch.cat([time_encodings, last_enc], dim=1)  # [B, T_i, time_dim]
                else:
                    padded_time_enc = time_encodings[:, :T_i, :]  # [B, T_i, time_dim]
                time_enc_reshaped = padded_time_enc.reshape(B * T_i, -1)

                # --- 改良2：处理动态图分支 (包含时间建模) ---
                if return_gate:
                    dynamic_out_reshaped, gate_reshaped = dynamic_conv(
                        features_reshaped, dynamic_adj_reshaped, time_enc_reshaped, features, window_size, return_gate=True
                    )
                    gate_broadcast = gate_reshaped.expand(-1, C, -1)
                    gate_unreshaped = gate_broadcast.view(B, T_i, C, D // 2)
                    head_gates.append(gate_unreshaped)
                else:
                    dynamic_out_reshaped = dynamic_conv(
                        features_reshaped, dynamic_adj_reshaped, time_enc_reshaped, features, window_size, return_gate=False
                    )

                # --- 改良2：处理静态图分支 (仅空间聚合) ---
                static_support = torch.bmm(static_adj_reshaped, features_reshaped)
                static_out_reshaped = static_conv(static_support)

                # --- 融合两个分支的输出 ---
                conv_out_reshaped = torch.cat([dynamic_out_reshaped, static_out_reshaped], dim=-1)
                conv_out = conv_out_reshaped.view(B, T_i, C, D) # 恢复形状

                # 上采样/下采样到基准时间步长 (base_time_steps)
                if T_i < base_time_steps:
                    conv_out = torch.nn.functional.interpolate(
                        conv_out.permute(0, 3, 2, 1),  # [B, D, C, T_i]
                        size=(num_channels, base_time_steps),
                        mode='nearest'
                    ).permute(0, 3, 2, 1)  # [B, T, C, D]
                    if return_gate:
                        gate_out = gate_broadcast.view(B, T_i, C, D)
                        gate_out = torch.nn.functional.interpolate( # type: ignore
                            gate_out.permute(0, 3, 2, 1),
                            size=(num_channels, base_time_steps),
                            mode='nearest'
                        ).permute(0, 3, 2, 1)
                        head_gates[-1] = gate_out
                elif T_i > base_time_steps:
                    conv_out = torch.nn.functional.adaptive_avg_pool1d(
                        conv_out.permute(0, 3, 2, 1),  # [B, D, C, T_i]
                        base_time_steps
                    ).permute(0, 3, 2, 1)  # [B, T, C, D]
                    if return_gate:
                        gate_out = gate_broadcast.view(B, T_i, C, D)
                        gate_out = torch.nn.functional.adaptive_avg_pool1d(
                            gate_out.permute(0, 3, 2, 1),
                            base_time_steps
                        ).permute(0, 3, 2, 1)
                        head_gates[-1] = gate_out

                head_outputs.append(conv_out)

            # 头融合
            if head_outputs:
                fused_output = self.head_fusion(torch.cat(head_outputs, dim=-1))
                multi_scale_outputs.append(fused_output)
                if return_gate:
                    # 对 gate 也做头融合（简单平均）
                    avg_gate_head = torch.stack(head_gates, dim=-1).mean(dim=-1)  # [B, T, C, D]
                    multi_scale_gates.append(avg_gate_head)

        # 多尺度融合
        if len(multi_scale_outputs) > 1:
            multi_scale_tensor = torch.stack(multi_scale_outputs, dim=1)  # [B, K, T, C, D]
            attn_weights = F.softmax(self.scale_attention[:len(multi_scale_outputs)], dim=0)
            final_output = torch.sum(multi_scale_tensor * attn_weights.view(1, -1, 1, 1, 1), dim=1)
            if return_gate:
                multi_scale_gate_tensor = torch.stack(multi_scale_gates, dim=1)
                avg_gate_weights = torch.sum(multi_scale_gate_tensor * attn_weights.view(1, -1, 1, 1, 1), dim=1)
            else:
                avg_gate_weights = None
        elif len(multi_scale_outputs) == 1:
            final_output = multi_scale_outputs[0]
            avg_gate_weights = multi_scale_gates[0] if return_gate else None
        else:
            final_output = features_list[0]
            avg_gate_weights = None

        output = self.layer_norm(features_list[0] + self.dropout(final_output))
        if return_gate:
            return output, avg_gate_weights
        else:
            return output


class AdaptiveTemporalGraphConvolution(nn.Module):
    """
    自适应时间窗口的图卷积 (优化版)
    核心思想: 将图卷积 (节点聚合) 与特征变换解耦。
    1. 图卷积: 仅在节点维度 (C) 上进行聚合，不改变特征维度 (D)。
    2. 特征变换: 使用独立的线性层对聚合后的特征进行升维/降维/非线性变换。
    """
    def __init__(self, in_features, out_features, time_features, max_window_size, args):
        super(AdaptiveTemporalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_window_size = max_window_size
        self.dgn_dr = args.dgn_dr

        # 空间信息处理 (图卷积)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # --- 改良2：引入标准的时间自注意力机制 ---
        self.temporal_attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=1, batch_first=True)
        self.temporal_norm = nn.LayerNorm(in_features)
        self.temporal_proj = nn.Linear(in_features, out_features)

        # 门控机制，融合时空信息
        self.adaptive_gate = nn.Sequential(
            nn.Linear(in_features, out_features), # 输入简化为节点自身特征
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(self.dgn_dr)

    def forward(self, features, adj_matrix, time_encoding, full_features, window_sizes, return_gate=False):
        """
        Args:
            features: [B*T, C, D] 当前时间步的特征
            adj_matrix: [B*T, C, C] 动态邻接矩阵
            time_encoding: [B*T, time_dim] 时间编码
            full_features: [B, T, C, D] 完整的历史序列特征，用于时间注意力
            window_sizes: [B, T, C] 自适应窗口大小
        """
        # 注意：这里的 batch_size 实际上是 B * T_current
        input_bt, num_channels, in_features = features.shape
        B, T_full, C, D = full_features.shape

        # 1. 空间信息聚合 (Graph Convolution)
        support = torch.bmm(adj_matrix, features)
        gcn_out = F.relu(torch.matmul(support, self.weight) + self.bias) # [B*T, C, D_out]

        # 2. 时间信息聚合 (Temporal Self-Attention)
        # full_features: [B, T, C, D] -> [B*C, T, D]
        temporal_in = rearrange(full_features, 'b t c d -> (b c) t d')
        
        # Query: 当前时间步 (取最后一个), Key/Value: 历史所有时间步
        q = temporal_in[:, -1:, :]
        k = v = temporal_in

        # === 构建自适应时间窗口 Mask ===
        if window_sizes.size(1) > 0:
            current_windows = window_sizes[:, -1, :].reshape(-1) # [B*C]
        else:
            current_windows = torch.ones(B*C, device=features.device) * self.max_window_size

        seq_indices = torch.arange(T_full, device=features.device).unsqueeze(0).expand(B * C, T_full)
        cutoff = (T_full - 1) - current_windows.unsqueeze(1)
        time_mask = seq_indices < cutoff # True 表示忽略

        temporal_out, _ = self.temporal_attention(q, k, v, key_padding_mask=time_mask) # [B*C, 1, D]
        temporal_out = self.temporal_norm(q + temporal_out)
        temporal_out = self.temporal_proj(temporal_out.squeeze(1)) # [B*C, D_out]
        
        # === 关键修复 Start ===
        # 计算当前 features 对应的时间步长度 target_T
        # features.shape[0] 是 B * target_T
        target_T = input_bt // B 
        
        # 1. 恢复到 [B, C, D_out]
        temporal_out = temporal_out.view(B, C, -1)
        
        # 2. 在时间维度扩展: [B, 1, C, D_out] -> [B, target_T, C, D_out]
        temporal_out = temporal_out.unsqueeze(1).expand(-1, target_T, -1, -1)
        
        # 3. 重新打平以匹配 GCN 输出: [B*target_T, C, D_out]
        # 务必使用 calculated shape，不要使用 input_bt * target_T
        temporal_out = temporal_out.reshape(input_bt, C, -1)
        # === 关键修复 End ===

        # 3. 门控融合
        gate_weight = self.adaptive_gate(features) # [B*T, C, D_out]
        final_output = gcn_out * gate_weight + temporal_out * (1 - gate_weight)
        final_output = self.dropout(final_output)

        if return_gate:
            return final_output, gate_weight
        else:
            return final_output
