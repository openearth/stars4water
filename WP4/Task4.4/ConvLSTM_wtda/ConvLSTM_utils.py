# hydrology_utils.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime
import pandas as pd

def calculate_api(precip, k):
    """Calculate Antecedent Precipitation Index"""
    T, Y, X = precip.shape
    api = np.zeros_like(precip, dtype=float)
    api[0, :, :] = precip[0, :, :]

    for t in range(1, T):
        api[t, :, :] = k * api[t - 1, :, :] + precip[t, :, :]

    return api

class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell implementation"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional layer for all gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute all gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        c = torch.tanh(cc_c)
        
        # Update cell state
        c_next = f * c_cur + i * c
        
        # Compute next hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden and cell states with zeros"""
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM model"""
    
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers=1, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.return_all_layers = return_all_layers
        
        # Create ConvLSTM layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    bias=bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: (batch, seq_len, channels, height, width)
            hidden_state: initial hidden state (optional)
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))
        
        layer_output_list = []
        layer_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            # Stack outputs along sequence dimension
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            layer_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            layer_state_list = layer_state_list[-1:]
        
        return layer_output_list, layer_state_list
    
    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

class HydrologyConvLSTM(nn.Module):
    """Simple ConvLSTM model for hydrological prediction"""
    
    def __init__(self, 
                 static_channels=3,     # topo, slopex, slopey
                 dynamic_channels=2,    # api, vwc
                 hidden_dim=64,
                 num_layers=2,
                 kernel_size=3):
        super(HydrologyConvLSTM, self).__init__()
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Conv2d(static_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Dynamic feature encoder
        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(dynamic_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTM(
            input_dim=hidden_dim * 2,  # static + dynamic features
            hidden_dims=[hidden_dim] * num_layers,
            kernel_sizes=[kernel_size] * num_layers,
            num_layers=num_layers,
            return_all_layers=False
        )
        
        # Decoder for prediction
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output: wtda prediction
        )
        
    def forward(self, static_data, dynamic_sequence):
        """
        Args:
            static_data: (batch, 3, H, W) - topo, slopex, slopey
            dynamic_sequence: (batch, seq_len, 2, H, W) - api, vwc time series
        
        Returns:
            prediction: (batch, 1, H, W) - predicted wtda
        """
        batch_size, seq_len = dynamic_sequence.shape[0], dynamic_sequence.shape[1]
        
        # 1. Process static features
        static_features = self.static_encoder(static_data)  # (batch, hidden_dim, H, W)
        
        # 2. Process dynamic features for each timestep
        dynamic_features_list = []
        for t in range(seq_len):
            dynamic_t = dynamic_sequence[:, t]  # (batch, 2, H, W)
            dynamic_features_t = self.dynamic_encoder(dynamic_t)  # (batch, hidden_dim, H, W)
            dynamic_features_list.append(dynamic_features_t)
        
        dynamic_features = torch.stack(dynamic_features_list, dim=1)  # (batch, seq_len, hidden_dim, H, W)
        
        # 3. Expand static features to match sequence and combine
        static_expanded = static_features.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        combined_features = torch.cat([static_expanded, dynamic_features], dim=2)
        
        # 4. Temporal modeling with ConvLSTM
        convlstm_outputs, _ = self.convlstm(combined_features)
        last_features = convlstm_outputs[-1][:, -1]  # Last layer, last timestep
        
        # 5. Decode to prediction
        prediction = self.decoder(last_features)
        
        return prediction

class SpatioTemporalDataset(Dataset):
    """Dataset for spatio-temporal hydrological data"""
    
    def __init__(self, static_data, dynamic_data, target_data, dates, seq_len=30):
        """
        Args:
            static_data: (C_static, H, W) - static features
            dynamic_data: (T, C_dynamic, H, W) - dynamic features over time
            target_data: (T, 1, H, W) - target values
            dates: array of datetime objects
            seq_len: length of input sequence
        """
        # Static: (C_static, H, W) - same for all samples
        self.static_data = torch.FloatTensor(static_data)
        
        # Dynamic and target: (T, channels, H, W)
        self.dynamic_data = torch.FloatTensor(dynamic_data)
        self.target_data = torch.FloatTensor(target_data)
        
        # Store dates
        self.date_indices = np.arange(len(dates))
        self.dates = dates  # Keep original dates for reference
        
        self.seq_len = seq_len
        self.total_timesteps = dynamic_data.shape[0]
        
    def __len__(self):
        # Number of possible sequences
        return max(0, self.total_timesteps - self.seq_len)
    
    def __getitem__(self, idx):
        # idx is the start time index (0 to total_timesteps-seq_len-1)
        
        # Static data is the same for all samples
        static = self.static_data  # (C_static, H, W)
        
        # Dynamic sequence: from idx to idx+seq_len
        dynamic_seq = self.dynamic_data[idx:idx+self.seq_len]  # (seq_len, C_dynamic, H, W)
        
        # Target: the NEXT timestep after the sequence
        target = self.target_data[idx+self.seq_len]  # (1, H, W)
        
        # Return date index
        target_idx = idx + self.seq_len
        
        return static, dynamic_seq, target, target_idx

def normalize_data(data, method='standard', mean=None, std=None, min_val=None, max_val=None):
    """Normalize data using either standard or min-max normalization"""
    if method == 'standard':
        if mean is None:
            mean = np.nanmean(data)
        if std is None:
            std = np.nanstd(data)
        normalized = (data - mean) / (std + 1e-8)
        return normalized, mean, std
    elif method == 'minmax':
        if min_val is None:
            min_val = np.nanmin(data)
        if max_val is None:
            max_val = np.nanmax(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized, min_val, max_val
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

def denormalize_wtda(data, wtda_std, wtda_mean):
    """Denormalize WTDA data"""
    return (data * wtda_std) + wtda_mean

def prepare_dates(dates_array):
    """Prepare dates array, decoding bytes if necessary"""
    if isinstance(dates_array[0], bytes):
        dates = [d.decode('utf-8') for d in dates_array]
        dates = np.array([datetime.strptime(d, '%Y-%m-%d') for d in dates])
    return dates

def calculate_normalization_stats(data_dict):
    """Calculate normalization statistics for multiple variables"""
    stats = {}
    for name, data in data_dict.items():
        stats[name] = {
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data)
        }
    return stats

def create_mask_from_data(data, nan_value=np.nan):
    """Create mask from data"""
    return np.where(np.isnan(data), nan_value, 1)