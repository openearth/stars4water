# Snow Water Equivalent (SWE) Prediction with ConvLSTM

This repository contains a deep learning pipeline for predicting Snow Water Equivalent (SWE) using an autoregressive Convolutional LSTM (ConvLSTM) model. The system integrates meteorological data (precipitation, temperature) with static geographical features (topography, latitude) to forecast SWE for 7 days ahead, with optimized GPU implementations.

## Key Features

- **Autoregressive ConvLSTM architecture** for spatiotemporal prediction
- **Multi-feature integration**: Combines dynamic (weather) and static (terrain) data
- **7-day forecasting** capability with scheduled sampling during training
- **GPU-optimized implementations** for both standard PyTorch and PyTorch with TRT (TensorRT)
- **Distributed training** support for large-scale datasets
- **HDF5 data handling** for efficient memory management

## File Structure

### Core Model Files

1. **snow_model.py**
   - Contains the core neural network architecture:
     - `ConvLSTMCell`: Basic building block for ConvLSTM operations
     - `AutoregressiveConvLSTM`: Main model class with autoregressive forecasting

2. **snow_data.py**
   - Data loading and preprocessing utilities:
     - `SweDataset`: PyTorch Dataset class for SWE prediction
     - `load_data`: Function to load and align input datasets
     - Data normalization and temporal encoding functions

### Training Scripts

3. **ConvLSTM_7days.py**
   - Main training script with:
     - Model initialization
     - Training loop with validation
     - Metric calculation (MAE, RMSE, R2)
     - Checkpoint saving

4. **ConvLSTM_7days_torchjsc.py**
   - Advanced version with:
     - Distributed training support
     - Gradient accumulation
     - Mixed precision training
     - HDF5 data handling
     - Memory optimization techniques

### Evaluation Scripts

5. **check_model.py**
   - Model evaluation and prediction script:
     - Loads trained model
     - Generates forecasts
     - Saves results as NetCDF files

6. **eval_CNNLSTM_weekly.py**
   - Extended evaluation script:
     - Handles different dataset configurations
     - Produces masked predictions
     - More comprehensive output formatting

## GPU Optimization Notes

The repository includes two specialized GPU implementations:

7. **python_GPU**:
   - Standard PyTorch CUDA implementation
   - Compatible with all CUDA-capable NVIDIA GPUs
   - Balanced performance and compatibility

8. **pythonjob_GPU_torchrun**:
   - PyTorch with TensorRT (TRT) implementation
   - Requires TensorRT installation
   - Provides maximum inference performance
   - Includes layer fusion and precision calibration

## Data Requirements

The model expects input data in NetCDF format with the following variables:

- **Dynamic variables** (time, lat, lon dimensions):
  - `snow_water_equivalent` (SWE in mm)
  - `rr` (precipitation)
  - `tg` (temperature)

- **Static variables** (lat, lon dimensions):
  - `dem_mean` (topography)
  - Latitude grid

