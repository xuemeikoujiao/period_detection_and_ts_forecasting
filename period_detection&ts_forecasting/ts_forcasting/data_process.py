import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import warnings

class MultiTimeSeriesDataset(Dataset):
    def __init__(self, df, input_size, forecast_size):
        """
        多时间序列数据集
        :param df: 包含三列的DataFrame [week_encode, se_ipvuv_1w, term]
        :param input_size: 输入序列长度
        :param forecast_size: 预测长度
        """
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.samples = []
        self.scalers = {}
        
        for term, group in df.groupby('term'):
            # 按周排序并处理缺失周
            full_series = self._complete_weeks(group)
            if full_series.empty:
                # print(f"Warning: term {term} has no data after completing weeks.")
                continue
            
            # 标准化
            scaler = Scaler()
            scaler.fit(full_series['se_ipvuv_1w'].values)
            scaled_series = scaler.transform(full_series['se_ipvuv_1w'].values)
            self.scalers[term] = scaler
            
            # 检查数据长度是否足够
            if len(scaled_series) <= input_size + forecast_size:
                print(f"Warning: term {term} has insufficient data for sliding window.")
                continue
            
            # 创建滑动窗口样本
            for i in range(len(scaled_series) - input_size - forecast_size + 1):
                self.samples.append({
                    'input': scaled_series[i:i+input_size],
                    'target': scaled_series[i+input_size:i+input_size+forecast_size],
                    'term': term
                })
        
        if not self.samples:
            raise ValueError("No samples were created. Check your input data or parameters.")

    def _complete_weeks(self, group):
        """ 补全缺失的周数据 """
        if group.empty:
            print("Warning: Group is empty.")
            return group
        min_week = group['week_encode'].min()
        max_week = group['week_encode'].max()
        all_weeks = pd.DataFrame({'week_encode': range(min_week, max_week + 1)})
        completed = pd.merge(all_weeks, group, on='week_encode', how='left').fillna(0)
        return completed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            t.FloatTensor(sample['input']),
            t.FloatTensor(sample['target']),
            sample['term']
        )
class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean

def create_dataloaders(df, input_size, forecast_size, batch_size=32, train_ratio=0.8):
    dataset = MultiTimeSeriesDataset(df, input_size, forecast_size)
    if len(dataset) == 0:
        raise ValueError("No samples were created. Check your input data or parameters.")
    train_size = int(len(dataset) * train_ratio)
    train_set, val_set = t.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader,val_loader, dataset.scalers

# 测试数据加载器

df = pd.read_csv("/Users/haochengzhang/Downloads/nbeats预测/test_data.csv")

# 设置参数
input_size = 2  # 使用过去2周的数据
forecast_size = 1  # 预测未来1周
batch_size = 32

# 创建数据加载器
# try:
train_loader,val_loader, scalers = create_dataloaders(df, input_size, forecast_size, batch_size)
for batch in train_loader:
    inputs, targets, terms = batch
    # print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}, terms={len(terms)}")
    # print(f"First term in batch: {terms[0]}")
    break
# except ValueError as e:
    # print(e)