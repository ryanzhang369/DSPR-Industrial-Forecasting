import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime

# =============================================================================
# 0. TimeMixer import check
# =============================================================================
try:
    from models.TimeMixer import Model as TimeMixer
    from utils.timefeatures import time_features as TimeFeatureEncoding
except ImportError:
    print("[Error] Cannot find models.TimeMixer or utils.timefeatures.")
    print("Please make sure you are using the standard directory structure of Time-Series-Library.")
    sys.exit(1)

# =============================================================================
# 1. TEP-specific physics loss function
# =============================================================================
class TEPPhysicsLoss(nn.Module):
    def __init__(self, scaler, col_map, device):
        super().__init__()
        self.device = device
        self.col_map = col_map
        # Convert scaler parameters to tensors
        self.means = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
        self.stds = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)

    def forward(self, preds_norm, inputs_norm):
        loss = 0.0

        # --- A. De-normalize predictions ---
        # Robust handling: if col_map key is missing, use a default index (-1)
        target_idx = self.col_map.get('Reactor_Pressure', -1)
        pred_phys = preds_norm * self.stds[target_idx] + self.means[target_idx]

        temp_idx = self.col_map.get('Reactor_Temp')

        # --- B. Physics constraints ---
        # 1) Non-negativity constraint
        loss_neg = torch.mean(torch.relu(-pred_phys))
        loss += loss_neg

        # 2) Dynamic smoothness (Total Variation)
        diff = pred_phys[:, 1:, :] - pred_phys[:, :-1, :]
        loss_tv = torch.mean(torch.abs(diff))
        loss += 0.5 * loss_tv

        # 3) Thermodynamic consistency (PV=nRT trend heuristic)
        if temp_idx is not None:
            temp_hist_norm = inputs_norm[:, :, temp_idx].unsqueeze(-1)
            temp_phys = temp_hist_norm * self.stds[temp_idx] + self.means[temp_idx]

            # Simple comparison: past temperature trend vs future predicted pressure trend
            delta_temp = temp_phys[:, -1, :] - temp_phys[:, -5, :]
            delta_pred = pred_phys[:, 5, :] - pred_phys[:, 0, :]

            trend_mismatch = -1.0 * torch.tanh(delta_temp) * torch.tanh(delta_pred)
            loss_trend = torch.mean(torch.relu(trend_mismatch))
            loss += 0.1 * loss_trend

        return loss

# =============================================================================
# 2. Model config and definitions
# =============================================================================
class TimeMixerConfigs:
    def __init__(self, args):
        # Copy all args fields into configs
        self.__dict__.update(vars(args))

        # Force-fix certain parameters to avoid logical conflicts
        self.use_future_temporal_feature = 0

        # Extra default values that some TimeMixer versions may require
        if not hasattr(self, 'output_attention'):
            self.output_attention = False
        if not hasattr(self, 'distil'):
            self.distil = True

class PGNN(nn.Module):
    def __init__(self, args):
        super(PGNN, self).__init__()
        self.args = args
        configs = TimeMixerConfigs(args)
        self.backbone = TimeMixer(configs)

    def forward(self, x_enc, x_mark_enc, u_future=None, x_mark_dec=None):
        B, T, C = x_enc.shape
        # TimeMixer requires dec_inp, usually zeros or partial history
        dec_inp = torch.zeros([B, self.args.pred_len, C], device=x_enc.device)

        output = self.backbone(x_enc, x_mark_enc, dec_inp, x_mark_dec)

        # In MS mode, only keep the last channel (target)
        if self.args.features == 'MS':
            return output[..., -1:]
        else:
            return output

# =============================================================================
# 3. Data processing
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, data, time_enc, seq_len, label_len, pred_len, features='MS'):
        self.data = data
        self.time_enc = time_enc
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]

        if self.features == 'M':
            seq_y = self.data[r_begin:r_end]
        else:
            seq_y = self.data[r_begin:r_end, -1:]

        mark_x = self.time_enc[s_begin:s_end]
        mark_y = self.time_enc[r_begin:r_end]

        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(mark_x, dtype=torch.float32),
                torch.tensor(mark_y, dtype=torch.float32))

def prepare_data(args, logger):
    abs_data_path = os.path.abspath(args.data_path)
    logger.info(f"[-] Loading data: {abs_data_path}")

    df = pd.read_csv(abs_data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # If there is no 'date' column, rename the first column to 'date'
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    feature_cols = [c for c in df.columns if c != 'date']
    target_name = args.target
    if target_name not in feature_cols:
        # Fallback: use the last column as target
        target_name = feature_cols[-1]

    input_cols = [c for c in feature_cols if c != target_name]
    # Ensure the target column is the last column
    sorted_cols = input_cols + [target_name]
    col_map = {name: i for i, name in enumerate(sorted_cols)}

    logger.info(f"[-] Feature column order: {sorted_cols}")
    logger.info(f"[-] Column index mapping: {col_map}")

    data_values = df[sorted_cols].values
    scaler = StandardScaler()

    total_len = len(df)
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)

    # Fit scaler only on the training portion
    scaler.fit(data_values[:train_len])
    data_scaled = scaler.transform(data_values)

    time_arr = TimeFeatureEncoding(pd.DatetimeIndex(df['date']), freq=args.freq).T

    # Build datasets (same underlying data; we split by index ranges below)
    train_set = CustomDataset(data_scaled, time_arr, args.seq_len, args.label_len, args.pred_len, args.features)
    val_set = CustomDataset(data_scaled, time_arr, args.seq_len, args.label_len, args.pred_len, args.features)
    test_set = CustomDataset(data_scaled, time_arr, args.seq_len, args.label_len, args.pred_len, args.features)

    # Split with Subset indices
    train_loader = DataLoader(
        Subset(train_set, range(0, train_len - args.seq_len - args.pred_len + 1)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        Subset(val_set, range(train_len, train_len + val_len - args.seq_len - args.pred_len + 1)),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(test_set, range(train_len + val_len, total_len - args.seq_len - args.pred_len + 1)),
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, sorted_cols, col_map

def metric(pred, true):
    # Align shapes if needed
    if pred.shape[1] != true.shape[1]:
        true = true[:, -pred.shape[1]:, :]

    MAE = np.mean(np.abs(pred - true))
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(true.flatten(), pred.flatten())
    return MAE, MSE, RMSE, 0, R2

def get_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('PGNN_TEP')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# =============================================================================
# 4. Main program
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    # --- Basic data parameters ---
    parser.add_argument('--task_name', type=str, default='long_term_forecast')  # newly added
    parser.add_argument('--data_path', type=str, default='tep.csv')
    parser.add_argument('--target', type=str, default='Reactor_Pressure')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='t')

    # --- Sequence length parameters ---
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # --- Training parameters ---
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--lambda_phys', type=float, default=0.1)

    # --- TimeMixer model parameters (extended) ---
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)  # add decoder layers to avoid errors
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--down_sampling_layers', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--use_norm', type=int, default=1)

    # --- TimeMixer-specific parameters ---
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--channel_independence', type=int, default=1)

    # --- Embedding-related parameters (to fix embedding errors) ---
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.save_dir, f"TEP_PGNN_{timestamp}")
    logger = get_logger(exp_dir)
    logger.info(args)

    train_loader, val_loader, test_loader, scaler, sorted_cols, col_map = prepare_data(args, logger)

    # Dynamically set input/output dimensions
    args.enc_in = len(sorted_cols)
    args.dec_in = len(sorted_cols)
    args.c_out = args.enc_in if args.features == 'M' else 1

    model = PGNN(args).to(args.device)

    criterion_mse = nn.MSELoss()
    criterion_phys = TEPPhysicsLoss(scaler, col_map, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    early_stop_cnt = 0

    for epoch in range(args.train_epochs):
        model.train()
        train_losses = []
        phys_losses = []

        for batch_x, batch_y, mark_x, mark_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            mark_x = mark_x.to(args.device)
            mark_y = mark_y.to(args.device)

            # Forward pass
            pred = model(batch_x, mark_x, x_mark_dec=mark_y)

            # Align labels to the prediction horizon
            true_pred = batch_y[:, -args.pred_len:, :]

            # Loss: data loss + weighted physics loss
            loss_data = criterion_mse(pred, true_pred)
            loss_phys = criterion_phys(pred, batch_x)
            loss = loss_data + args.lambda_phys * loss_phys

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            phys_losses.append(loss_phys.item())

        avg_train = np.mean(train_losses)
        avg_phys = np.mean(phys_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, mark_x, mark_y in val_loader:
                batch_x = batch_x.to(args.device)
                batch_y = batch_y.to(args.device)
                mark_x = mark_x.to(args.device)
                mark_y = mark_y.to(args.device)

                pred = model(batch_x, mark_x, x_mark_dec=mark_y)
                true_pred = batch_y[:, -args.pred_len:, :]
                val_losses.append(criterion_mse(pred, true_pred).item())

        avg_val = np.mean(val_losses)
        logger.info(f"Epoch {epoch+1} | Train: {avg_train:.5f} (Phys: {avg_phys:.5f}) | Val: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_cnt = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                logger.info("Early stopping.")
                break

    # ==========================================
    # 5. Final testing
    # ==========================================
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch_x, batch_y, mark_x, mark_y in test_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            mark_x = mark_x.to(args.device)
            mark_y = mark_y.to(args.device)

            pred = model(batch_x, mark_x, x_mark_dec=mark_y)

            preds.append(pred.cpu().numpy())
            # Note: labels are still in normalized space here
            trues.append(batch_y[:, -args.pred_len:, :].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # -------------------------------------------------------------------------
    # Evaluation in normalized space (Normalized metrics)
    # -------------------------------------------------------------------------
    mae_norm, mse_norm, rmse_norm, _, r2_norm = metric(preds, trues)
    logger.info(f"\n[Test Result] >>> Normalized Scale <<<")
    logger.info(f"MAE : {mae_norm:.4f}")
    logger.info(f"RMSE: {rmse_norm:.4f}")
    logger.info(f"R2  : {r2_norm:.4f}")

    # -------------------------------------------------------------------------
    # Evaluation in original space (Original scale metrics)
    # -------------------------------------------------------------------------
    target_idx = col_map[args.target]
    mean = scaler.mean_[target_idx]
    std = scaler.scale_[target_idx]

    preds_orig = preds * std + mean
    trues_orig = trues * std + mean

    mae_orig, mse_orig, rmse_orig, _, r2_orig = metric(preds_orig, trues_orig)
    logger.info(f"\n[Test Result] >>> Original Scale ({args.target}) <<<")
    logger.info(f"MAE : {mae_orig:.4f}")
    logger.info(f"RMSE: {rmse_orig:.4f}")
    logger.info(f"R2  : {r2_orig:.4f}")

    # Save both normalized and original-scale results for further analysis
    np.savez(
        os.path.join(exp_dir, 'test_results.npz'),
        preds_norm=preds, trues_norm=trues,
        preds_orig=preds_orig, trues_orig=trues_orig
    )

if __name__ == "__main__":
    main()
