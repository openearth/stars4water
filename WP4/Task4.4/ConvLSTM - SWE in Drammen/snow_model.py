# snow_model.py
import torch
import torch.nn as nn
import numpy as np

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell"""
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
    """Autoregressive ConvLSTM Model for Snow Prediction"""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, static_channels):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.static_channels = static_channels
        
        # Store configuration parameters
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'kernel_size': kernel_size,
            'num_layers': num_layers,
            'static_channels': static_channels
        }
        
        # Network components
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size) 
            for _ in range(num_layers)
        ])
        self.static_processor = nn.Sequential(
            nn.Conv2d(static_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        self.weather_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.temp_proj = nn.Linear(4, hidden_dim)
        self.step_proj = nn.Conv2d(1 + hidden_dim * 3, hidden_dim, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, static, future_precip, future_temp, future_temporal, target=None, epoch=0):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process features
        static_feat = self.static_processor(static)
        h = [torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
            for _ in range(self.num_layers)]
        c = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = self.input_conv(x[:, t])
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_t, h[i], c[i])
                x_t = h[i]
        
        # Autoregressive prediction
        predictions = []
        last_swe = x[:, -1, 0:1]
        
        for step in range(7):  # FORECAST_STEPS = 7
            weather_t = torch.stack([future_precip[:, step], future_temp[:, step]], dim=1)
            weather_feat = self.weather_proj(weather_t)
            temporal_feat = self.temp_proj(future_temporal[:, step]).view(batch_size, -1, 1, 1).expand(-1, -1, height, width)
            
            combined = torch.cat([last_swe, weather_feat, static_feat, temporal_feat], dim=1)
            x_next = self.step_proj(combined)
            
            for i, layer in enumerate(self.convlstm_layers):
                h[i], c[i] = layer(x_next, h[i], c[i])
                x_next = h[i]
            
            next_swe = self.conv_out(x_next)
            predictions.append(next_swe)
            
            if self.training and target is not None:
                use_gt = torch.rand(1).item() < (0.5 * (1 - epoch/50))
                last_swe = target[:, step:step+1] if use_gt else next_swe.detach()
            else:
                last_swe = next_swe
        
        return torch.cat(predictions, dim=1)