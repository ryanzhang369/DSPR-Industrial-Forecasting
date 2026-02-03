import argparse
import logging
import os
import sys
import time
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime
from tqdm import tqdm

# --- 核心模型导入 ---
try:
    from models.TimeMixer import Model as TimeMixer
    from layers.dyna_explain_graph_constructor import AdaptiveDynamicGraphConstructor
    from layers.dyna_explain_graph_layer import AdaptiveWindowDGNNLayer
    from utils.timefeatures import time_features as TimeFeatureEncoding
except ImportError:
    print("错误: 未找到模型文件。请确保 models/, layers/, utils/ 目录存在。")
    sys.exit(1)

# =============================================================================
# 0. 工具函数
# =============================================================================
def fix_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir):
    """双向日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('DSPR_Experiment')
    logger.setLevel(logging.INFO)
    logger.handlers = [] 
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger

def metric(pred, true):
    MAE = np.mean(np.abs(pred - true))
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((pred - true) / (true + 1e-5))) 
    R2 = r2_score(true.flatten(), pred.flatten())
    return MAE, MSE, RMSE, MAPE, R2

# =============================================================================
# 1. 物理图加载器
# =============================================================================
class PhysicsLoader:
    @staticmethod
    def load_adjacency_matrix(adj_path, model_columns, device, logger):
        num_vars = len(model_columns)
        if not adj_path or not os.path.exists(adj_path):
            logger.info(f"[Physics] 未检测到物理图文件或路径为空，使用纯数据驱动模式。")
            return torch.zeros((num_vars, num_vars), dtype=torch.float32).to(device)

        logger.info(f"[Physics] 加载物理先验图: {adj_path}")
        try:
            adj_df = pd.read_csv(adj_path, index_col=0)
        except Exception as e:
            logger.error(f"[Error] 物理图读取失败: {e}")
            return torch.zeros((num_vars, num_vars), dtype=torch.float32).to(device)

        adj_tensor = torch.zeros((num_vars, num_vars), dtype=torch.float32)
        match_count = 0
        for i, source_name in enumerate(model_columns):
            for j, target_name in enumerate(model_columns):
                if source_name in adj_df.index and target_name in adj_df.columns:
                    weight = adj_df.loc[source_name, target_name]
                    if pd.notna(weight) and float(weight) != 0:
                        adj_tensor[i, j] = float(weight)
                        match_count += 1
        
        logger.info(f"[Physics] 成功映射 {match_count} 条物理边。")
        if adj_tensor.max() > 0:
            adj_tensor = adj_tensor / (adj_tensor.max() + 1e-8)
        return adj_tensor.to(device)

# =============================================================================
# 2. DSPR 模型定义
# =============================================================================
class TimeMixerConfigs:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.task_name = 'long_term_forecast'
        self.use_future_temporal_feature = 1

class DSPR(nn.Module):
    def __init__(self, args, physical_adj=None):
        super(DSPR, self).__init__()
        self.args = args
        
        # 物理先验
        if physical_adj is not None:
            self.register_buffer("phys_prior", physical_adj)
            self.register_buffer("phys_mask", (physical_adj > 0).float())
        else:
            self.phys_prior = None
            self.phys_mask = None

        # GNN 组件
        self.channel_embeddings = nn.ModuleList([
            nn.Linear(1, args.d_model) for _ in range(args.enc_in)
        ])
        self.embedding_norm = nn.LayerNorm(args.d_model)

        self.graph_constructor = AdaptiveDynamicGraphConstructor(
            input_dim=args.d_model + args.time_dim,
            num_channels=args.enc_in,
            d_model=args.d_model,
            max_window_size=20
        )
        self.dgnn_layer = AdaptiveWindowDGNNLayer(
            d_model=args.d_model,
            time_dim=args.time_dim,
            max_window_size=20,
            args=args,
            num_heads=args.gnn_heads
        )
        self.final_norm = nn.LayerNorm(args.d_model)

        # 残差投影
        self.delta_proj = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.d_model * 2, args.pred_len)
        )
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

        # Backbone
        configs = TimeMixerConfigs(args)
        configs.c_out = args.c_out 
        self.backbone = TimeMixer(configs)
        
        # 融合系数
        self.alpha_logit = nn.Parameter(torch.tensor(-10.0)) 
        self.alpha = configs.phys_alpha 

    def fuse_with_physical_prior(self, learned_adj):
        if self.phys_prior is None: return learned_adj
        phys = self.phys_prior
        mask = self.phys_mask
        while phys.dim() < learned_adj.dim():
            phys = phys.unsqueeze(0)
            mask = mask.unsqueeze(0)
        fused = (1 - mask) * learned_adj + mask * (
            (1 - self.alpha) * learned_adj + self.alpha * phys 
        )
        fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-6)
        return fused

    def forward(self, x_enc, x_mark_enc, u_future=None, x_mark_dec=None, return_extra=False):
        B, T, C = x_enc.shape
        
        # 1. Backbone
        x_dec_dummy = torch.zeros([B, self.args.pred_len, C], device=x_enc.device)
        tm_out = self.backbone(x_enc, x_mark_enc, x_dec_dummy, x_mark_dec)
        
        if self.args.features == 'MS':
            main_pred = tm_out[..., -1:] 
        else:
            main_pred = tm_out 

        # 2. GNN Stream
        embedded_list = [self.channel_embeddings[c](x_enc[:, :, c:c+1]).unsqueeze(2) for c in range(C)]
        x_embed = self.embedding_norm(torch.cat(embedded_list, dim=2))
       
        static_adj, dynamic_adj, masks, win = self.graph_constructor([x_embed], x_mark_enc)
        
        if dynamic_adj is not None and getattr(self, "phys_prior", None) is not None:
            dynamic_adj = [self.fuse_with_physical_prior(adj) for adj in dynamic_adj]

        graph_out = self.dgnn_layer([x_embed], static_adj, dynamic_adj, masks, win, x_mark_enc)[0]
        enhanced = self.final_norm(graph_out + x_embed)
        
        # 3. Residual
        if self.args.features == 'MS':
            target_feat = enhanced[:, :, -1, :].mean(dim=1) 
            residual = self.delta_proj(target_feat).unsqueeze(-1)
        else:
            residual = 0 

        final_pred = main_pred + torch.sigmoid(self.alpha_logit) * residual

        if return_extra:
            return final_pred, dynamic_adj, win
        return final_pred

# =============================================================================
# 3. 数据处理
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, data, time_enc, seq_len, pred_len, u_idx, features='MS'):
        self.data = data
        self.time_enc = time_enc
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.u_idx = u_idx
        self.features = features
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        
        seq_x = self.data[index:s_end]
        if self.features == 'M':
            seq_y = self.data[s_end:r_end]
        else:
            seq_y = self.data[s_end:r_end, -1:]
        seq_u = self.data[s_end:r_end, self.u_idx:self.u_idx+1]
        mark_x = self.time_enc[index:s_end]
        mark_y = self.time_enc[s_end:r_end]
        
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(mark_x, dtype=torch.float32),
                torch.tensor(mark_y, dtype=torch.float32),
                torch.tensor(seq_u, dtype=torch.float32))

def prepare_data(args, logger):
    logger.info(f"[-] 读取数据: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")

    df = pd.read_csv(args.data_path)
    if 'date' not in df.columns:
        if 'time' in df.columns: df.rename(columns={'time': 'date'}, inplace=True)
        else: df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    cols = [c for c in df.columns if c != 'date']
    if args.target not in cols:
        logger.warning(f"目标列 {args.target} 未找到，尝试使用最后一列作为 Target")
        args.target = cols[-1]
    
    feature_cols = [c for c in cols if c != args.target]
    sorted_cols = feature_cols + [args.target]
    
    logger.info(f"[-] 模式: {args.features}")
    logger.info(f"[-] 特征数量: {len(sorted_cols)} | Target: {args.target}")

    u_idx = 0
    if args.control_col and args.control_col in sorted_cols:
        u_idx = sorted_cols.index(args.control_col)
    
    data_values = df[sorted_cols].values
    scaler = StandardScaler()
    
    total = len(df)
    train_len = int(0.7 * total)
    val_len = int(0.1 * total)
    
    scaler.fit(data_values[:train_len])
    scaled_data = scaler.transform(data_values)
    
    time_arr = TimeFeatureEncoding(pd.DatetimeIndex(df['date']), freq=args.freq).T
    
    train_set = CustomDataset(scaled_data, time_arr, args.seq_len, args.pred_len, u_idx, args.features)
    val_set = CustomDataset(scaled_data, time_arr, args.seq_len, args.pred_len, u_idx, args.features)
    test_set = CustomDataset(scaled_data, time_arr, args.seq_len, args.pred_len, u_idx, args.features)
    
    train_loader = DataLoader(Subset(train_set, range(0, train_len - args.seq_len - args.pred_len + 1)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(val_set, range(train_len, train_len + val_len - args.seq_len - args.pred_len + 1)), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_set, range(train_len + val_len, total - args.seq_len - args.pred_len + 1)), batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, sorted_cols

# =============================================================================
# 4. 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='DSPR Runner')
    
    # 基础设置
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--features', type=str, default='MS', choices=['M', 'MS'])
    parser.add_argument('--adj_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./records/')

    # 序列参数
    parser.add_argument('--control_col', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--freq', type=str, default='t')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--phys_alpha', type=float, default=0.5)
    parser.add_argument('--lambda_phys', type=float, default=0.01)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--gnn_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--dgn_dr', type=float, default=0.05)
    parser.add_argument('--time_dim', type=int, default=5)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--down_sampling_layers', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)

    args = parser.parse_args()

    fix_seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.save_dir, f"{args.run_name}_{timestamp}")
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    res_dir = os.path.join(exp_dir, 'results')
    
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    
    logger = get_logger(exp_dir)
    logger.info(f"DSPR Experiment Started | Seed: {args.seed} | Device: cuda:{args.gpu}")
    logger.info(args)

    args.data_path = os.path.join(args.root_path, args.data_path)
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, scaler, sorted_cols = prepare_data(args, logger)
    args.enc_in = len(sorted_cols)
    args.dec_in = len(sorted_cols)
    args.c_out = args.enc_in if args.features == 'M' else 1
    
    phys_adj = PhysicsLoader.load_adjacency_matrix(args.adj_path, sorted_cols, args.device, logger)
    model = DSPR(args, phys_adj).to(args.device)
    
    # -------------------------------------------------------------------------
    # 核心修改：使用标准 Adam 优化器 + ReduceLROnPlateau 调度器
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # 调度器：当 Validation Loss 连续 1 个 Epoch 不下降时，学习率减半 (factor=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=1
    )

    criterion = nn.HuberLoss(delta=1.0)
    
    logger.info(f"[-] Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =======================================================
    # 训练循环
    # =======================================================
    best_val_loss = float('inf')
    early_stop_count = 0
    
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.train_epochs}", leave=True)
        
        for batch_x, batch_y, mark_x, mark_y, u in train_pbar:
            optimizer.zero_grad()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            mark_x = mark_x.to(args.device)
            mark_y = mark_y.to(args.device)
            
            pred, d_adj, _ = model(batch_x, mark_x, u, mark_y, return_extra=True)
            loss = criterion(pred, batch_y)
            
            if d_adj is not None and model.phys_prior is not None:
                curr_adj = d_adj[0]
                phys_loss = (((curr_adj - model.phys_prior)**2) * model.phys_mask).mean()
                loss += args.lambda_phys * phys_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            
            train_loss.append(loss.item())
            train_pbar.set_postfix({'loss': f"{loss.item():.5f}"})
        
        # Validation
        model.eval()
        val_loss = []
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_x, batch_y, mark_x, mark_y, u in val_loader:
                batch_x = batch_x.to(args.device)
                batch_y = batch_y.to(args.device)
                mark_x = mark_x.to(args.device)
                mark_y = mark_y.to(args.device)
                pred = model(batch_x, mark_x, u, mark_y)
                val_loss.append(criterion(pred, batch_y).item())
        
        avg_train = np.mean(train_loss)
        avg_val = np.mean(val_loss)
        
        # 调度器在 Validation 后 Update
        scheduler.step(avg_val)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}: Train Loss {avg_train:.5f} | Val Loss {avg_val:.5f} | LR {current_lr:.7f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_count = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint.pth'))
            logger.info("  --> Best Model Saved.")
        else:
            early_stop_count += 1
            if early_stop_count >= args.patience:
                logger.info("[-] Early Stopping triggered.")
                break

    # =======================================================
    # 测试
    # =======================================================
    logger.info("[-] 开始测试...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
    model.eval()
    
    preds = []
    trues = []
    learned_adjs = [] 
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_x, batch_y, mark_x, mark_y, u in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            mark_x = mark_x.to(args.device)
            mark_y = mark_y.to(args.device)
            
            pred, d_adj, _ = model(batch_x, mark_x, u, mark_y, return_extra=True)
            preds.append(pred.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            
            if d_adj is not None:
                adj_snapshot = d_adj[0].mean(dim=0).mean(dim=0).cpu().numpy()
                learned_adjs.append(adj_snapshot)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    final_adj = np.mean(learned_adjs, axis=0) if len(learned_adjs) > 0 else np.zeros((args.enc_in, args.enc_in))

    # Metrics
    mae, mse, rmse, mape, r2 = metric(preds, trues)
    logger.info(f"\n[Evaluation - Scaled Space] MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    if args.features == 'MS':
        target_idx = sorted_cols.index(args.target)
        mean_val = scaler.mean_[target_idx]
        std_val = scaler.scale_[target_idx]
        preds_orig = preds * std_val + mean_val
        trues_orig = trues * std_val + mean_val
    else:
        shape_p = preds.shape
        preds_orig = scaler.inverse_transform(preds.reshape(-1, shape_p[-1])).reshape(shape_p)
        trues_orig = scaler.inverse_transform(trues.reshape(-1, shape_p[-1])).reshape(shape_p)

    mae_o, mse_o, rmse_o, mape_o, r2_o = metric(preds_orig, trues_orig)
    logger.info(f"[Evaluation - Original Space] MSE: {mse_o:.4f} | MAE: {mae_o:.4f} | RMSE: {rmse_o:.4f}")

    np.savez(
        os.path.join(res_dir, 'results.npz'), 
        preds=preds, trues=trues,
        preds_orig=preds_orig, trues_orig=trues_orig,
        adj_matrix=final_adj
    )
    logger.info(f"[-] 实验结束。结果已归档至: {exp_dir}")

if __name__ == '__main__':
    main()