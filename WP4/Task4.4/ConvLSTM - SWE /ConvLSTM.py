
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SweDataset(Dataset):
    def __init__(self, swe, precip, temp, static, time_indices, times, seq_len, forecast_steps, stats=None):
        self.swe = np.nan_to_num(np.log1p(swe), nan=0.0)  # log1p transform
        self.precip = np.nan_to_num(precip, nan=0.0)
        self.temp = np.nan_to_num(temp, nan=0.0)
        self.static = np.nan_to_num(static, nan=0.0)
        self.time_indices = time_indices
        self.stats = stats if stats else self._compute_stats()
        self.times = times
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps

    def _compute_stats(self):
        return {
            'swe_min': np.nanmin(self.swe),
            'swe_max': np.nanmax(self.swe),
            'precip_min': np.nanmin(self.precip),
            'precip_max': np.nanmax(self.precip),
            'temp_mean': np.nanmean(self.temp),
            'temp_std': np.nanstd(self.temp),
            'topo_min': np.nanmin(self.static[0]),
            'topo_max': np.nanmax(self.static[0]),
            'lat_mean': np.nanmean(self.static[1]),
            'lat_std': np.nanstd(self.static[1])
        }

    def _normalize_dynamic(self, data, feature):
        if feature == 'swe':
            return (data - self.stats['swe_min']) / (self.stats['swe_max'] - self.stats['swe_min'])
        elif feature == 'precip':
            return (data - self.stats['precip_min']) / (self.stats['precip_max'] - self.stats['precip_min'])
        elif feature == 'temp':
            return (data - self.stats['temp_mean']) / self.stats['temp_std']
        else:
            raise ValueError(f"Unknown feature: {feature}")

    def _normalize_static(self):
        topo = (self.static[0] - self.stats['topo_min']) / (self.stats['topo_max'] - self.stats['topo_min'])
        lat = (self.static[1] - self.stats['lat_mean']) / self.stats['lat_std']
        return np.stack([topo, lat])

    def __len__(self):
        return len(self.time_indices) - self.seq_len - self.forecast_steps

    def __getitem__(self, idx):
        actual_t = self.time_indices[idx + self.seq_len]

        # 1. SWE: Past 60 days (t-60 to t-1)
        swe_past = self._normalize_dynamic(
            self.swe[actual_t - self.seq_len : actual_t],
            'swe'
        )

        # 2. Precipitation: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        precip_seq = self._normalize_dynamic(
            self.precip[actual_t - self.seq_len : actual_t + self.forecast_steps],
            'precip'
        )

        # 3. Temperature: Past 60 days (t-60 to t-1) + Future 7 days (t to t+6)
        temp_seq = self._normalize_dynamic(
            self.temp[actual_t - self.seq_len : actual_t + self.forecast_steps],
            'temp'
        )

        # 4. Temporal encoding for ALL timesteps (history + future)
        all_days = np.arange(actual_t - self.seq_len, actual_t + self.forecast_steps)
        all_months = ((all_days % 365) // 30) + 1  # Approximate month
        all_doy = (all_days % 365) + 1  # Day of year
        
        # Cyclical encoding
        month_sin = np.sin(2 * np.pi * all_months / 12)
        month_cos = np.cos(2 * np.pi * all_months / 12)
        day_sin = np.sin(2 * np.pi * all_doy / 365)
        day_cos = np.cos(2 * np.pi * all_doy / 365)

        # Split into historical and future parts
        hist_month_sin = month_sin[:self.seq_len]
        hist_month_cos = month_cos[:self.seq_len]
        hist_day_sin = day_sin[:self.seq_len]
        hist_day_cos = day_cos[:self.seq_len]
        
        fut_month_sin = month_sin[self.seq_len:]
        fut_month_cos = month_cos[self.seq_len:]
        fut_day_sin = day_sin[self.seq_len:]
        fut_day_cos = day_cos[self.seq_len:]

        # Ensure all historical arrays have shape (SEQ_LEN, height, width)
        height, width = swe_past.shape[1], swe_past.shape[2]
        
        # Expand temporal features to match spatial dimensions
        hist_month_sin = np.tile(hist_month_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_month_cos = np.tile(hist_month_cos[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_sin = np.tile(hist_day_sin[:, np.newaxis, np.newaxis], (1, height, width))
        hist_day_cos = np.tile(hist_day_cos[:, np.newaxis, np.newaxis], (1, height, width))
        
        forecast_dates = self.times[actual_t : actual_t + self.forecast_steps]

        # 5. Stack all historical dynamic features
        dynamic = np.stack([
            swe_past,                         # Channel 0: SWE (60 steps)
            precip_seq[:self.seq_len],        # Channel 1: Past precip (60 steps)
            temp_seq[:self.seq_len],          # Channel 2: Past temp (60 steps)
            hist_month_sin,                   # Channel 3: Month (sin)
            hist_month_cos,                   # Channel 4: Month (cos)
            hist_day_sin,                     # Channel 5: Day of year (sin)
            hist_day_cos                      # Channel 6: Day of year (cos)
        ], axis=1)  # Shape: (60, 7, height, width)

        # 6. Future weather (t to t+6)
        future_precip = precip_seq[self.seq_len:]  # Shape: (7, height, width)
        future_temp = temp_seq[self.seq_len:]      # Shape: (7, height, width)

        # 7. Future temporal features (t to t+6)
        future_temporal = np.stack([
            fut_month_sin,
            fut_month_cos,
            fut_day_sin,
            fut_day_cos
        ], axis=1)  # Shape: (7, 4)

        # 8. Static features
        static = self._normalize_static()  # Shape: (2, height, width)

        # 9. Target: SWE for next 7 days (t to t+6)
        target = self._normalize_dynamic(
            self.swe[actual_t : actual_t + self.forecast_steps],
            'swe'
        )  # Shape: (7, height, width)

        return (
            torch.tensor(dynamic, dtype=torch.float32),    # (60, 7, H, W)
            torch.tensor(static, dtype=torch.float32),     # (2, H, W)
            torch.tensor(future_precip, dtype=torch.float32),  # (7, H, W)
            torch.tensor(future_temp, dtype=torch.float32),    # (7, H, W)
            torch.tensor(future_temporal, dtype=torch.float32), # (7, 4)
            torch.tensor(target, dtype=torch.float32),
            forecast_dates # (7, H, W)
        )

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class AutoregressiveConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, static_channels):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Initial layer to process dynamic input
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                hidden_dim,
                hidden_dim,
                kernel_size
            ) for _ in range(num_layers)
        ])
        
        # Static feature processing
        self.static_processor = nn.Sequential(
            nn.Conv2d(static_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        # Weather feature processing
        self.weather_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        
        # Temporal feature projection
        self.temp_proj = nn.Linear(4, hidden_dim)
        
        # Projection for autoregressive step
        self.step_proj = nn.Conv2d(1 + hidden_dim + hidden_dim + hidden_dim, hidden_dim, kernel_size=1)
        
        # Output layer
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, *args, **kwargs):
        # Handle DataParallel input wrapping
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        
        # Unpack arguments
        if len(args) >= 5:
            x, static, future_precip, future_temp, future_temporal = args[:5]
            target = kwargs.get('target', None)
            epoch = kwargs.get('epoch', 0)
        else:
            x = kwargs['x']
            static = kwargs['static']
            future_precip = kwargs['future_precip']
            future_temp = kwargs['future_temp']
            future_temporal = kwargs['future_temporal']
            target = kwargs.get('target', None)
            epoch = kwargs.get('epoch', 0)

        batch_size, seq_len, channels, height, width = x.shape
        
        # Process static features
        static_feat = self.static_processor(static)
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
            for _ in range(self.num_layers)]
        c = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
        
        # Process historical sequence
        for t in range(seq_len):
            x_t = self.input_conv(x[:, t])
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_t, h[i], c[i])
                x_t = h[i]
        
        predictions = []
        last_swe = x[:, -1, 0:1]  # Last SWE value (channel 0)
        
        # Autoregressive prediction
        for step in range(FORECAST_STEPS):
            # Process weather features
            weather_t = torch.stack([
                future_precip[:, step], 
                future_temp[:, step]
            ], dim=1)
            weather_feat = self.weather_proj(weather_t)
            
            # Process temporal features
            temporal_feat = self.temp_proj(future_temporal[:, step])  # [B, hidden_dim]
            temporal_feat = temporal_feat.view(batch_size, self.hidden_dim, 1, 1).expand(-1, -1, height, width)
            
            # Combine all features
            combined = torch.cat([
                last_swe,                    # 1 channel
                weather_feat,                # hidden_dim channels
                static_feat,                 # hidden_dim channels
                temporal_feat                # hidden_dim channels
            ], dim=1)
            
            x_next = self.step_proj(combined)
            
            # Process through ConvLSTM
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_next, h[i], c[i])
                x_next = h[i]
            
            # Predict next SWE
            next_swe = self.conv_out(x_next)
            predictions.append(next_swe)
            
            # Scheduled sampling
            if self.training and target is not None:
                use_gt = torch.rand(1).item() < (0.5 * (1 - epoch/EPOCHS))
                last_swe = target[:, step:step+1] if use_gt else next_swe.detach()
            else:
                last_swe = next_swe
        
        return torch.cat(predictions, dim=1)

def denormalize_swe(normalized_swe, stats):
    log_swe = normalized_swe * (stats['swe_max'] - stats['swe_min']) + stats['swe_min']
    swe = np.expm1(log_swe)  
    return swe


class BiasAwareLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        bias = torch.mean(pred - target)
        return mse_loss + self.alpha * bias**2

def calculate_metrics(output, target, stats):

    output_denorm = torch.expm1(output * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    target_denorm = torch.expm1(target * (stats['swe_max'] - stats['swe_min']) + stats['swe_min'])
    
    # Calculate metrics
    mae = torch.mean(torch.abs(output_denorm - target_denorm))
    mse = torch.mean((output_denorm - target_denorm)**2)
    rmse = torch.sqrt(mse)
    
    # Calculate R2 score
    target_mean = torch.mean(target_denorm)
    ss_total = torch.sum((target_denorm - target_mean)**2)
    ss_res = torch.sum((output_denorm - target_denorm)**2)
    r2 = 1 - (ss_res / ss_total)
    
    return mae, rmse, r2