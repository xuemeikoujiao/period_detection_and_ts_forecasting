# N-BEATS Time Series Forecasting

An implementation of N-BEATS model for time series forecasting.

## Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy

## Usage

1. Prepare data in CSV format (single column time series)
2. Run training:
```bash
python run.py \
  --data_path your_data.csv \
  --input_size 100 \
  --forecast_size 7 \
  --batch_size 32 \
  --epochs 50