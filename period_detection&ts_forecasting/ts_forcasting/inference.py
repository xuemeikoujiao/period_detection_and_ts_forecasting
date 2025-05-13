import numpy as np
import torch as t
from model import create_model
import matplotlib.pyplot as plt
def predict(model, x, device):
    model.eval()
    with t.no_grad():
        x = t.FloatTensor(x).unsqueeze(0).to(device)
        return model(x, t.ones_like(x)).cpu().numpy().squeeze()

# def smape(y_true, y_pred):
#     return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
def smape(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, y_pred.shape[1])  # 调整 y_true 的形状
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


# def evaluate_model(model, test_loader, device, scaler, plot_results=False):
#     model.eval()
#     predictions, trues = [], []
#     with t.no_grad():
#         for x, y, terms in test_loader:  # 获取 terms
#             x, y = x.to(device), y.to(device)
#             pred = model(x, t.ones_like(x)).cpu().numpy()
            
#             # 根据 term 获取对应的 Scaler
#             for i, term in enumerate(terms):
#                 term = term.item() if isinstance(term, t.Tensor) else term
#                 pred[i] = scaler[term].inverse_transform(pred[i])
#                 true = scaler[term].inverse_transform(y[i].cpu().numpy())  # 转换为 numpy
#                 trues.append(true.reshape(1, -1))  # 保留二维形状
            
#             predictions.append(pred)
    
#     # 确保 predictions 和 trues 的形状一致
#     predictions = np.concatenate(predictions, axis=0)  # 拼接时保持二维形状
#     trues = np.concatenate(trues, axis=0)  # 拼接时保持二维形状
#     print(f"Final predictions shape: {predictions.shape}")
#     print(f"Final trues shape: {trues.shape}")
#     # 绘制真实值和预测值的曲线
#     if plot_results:
#         plt.figure(figsize=(12, 6))
#         for i in range(min(5, len(predictions))):  # 绘制前 5 个样本
#             plt.plot(trues[i], label=f"True {i+1}", linestyle='--')
#             plt.plot(predictions[i], label=f"Pred {i+1}")
#         plt.title("True vs Predicted Values")
#         plt.xlabel("Time Steps")
#         plt.ylabel("Values")
#         plt.legend()
#         plt.show()
#     return smape(trues, predictions), np.sqrt(np.mean((trues - predictions)**2))
def evaluate_model(model, test_loader, device, scaler, plot_results=False):
    model.eval()
    predictions, trues, backcasts = [], [], []
    with t.no_grad():
        for x, y, terms in test_loader:  # 获取 terms
            x, y = x.to(device), y.to(device)
            backcast, forecast = model(x, t.ones_like(x))  # 获取 backcast 和 forecast
            pred = forecast.cpu().numpy()
            backcast = backcast.cpu().numpy()
            
            # 根据 term 获取对应的 Scaler
            for i, term in enumerate(terms):
                term = term.item() if isinstance(term, t.Tensor) else term
                pred[i] = scaler[term].inverse_transform(pred[i])
                backcast[i] = scaler[term].inverse_transform(backcast[i])
                true = scaler[term].inverse_transform(y[i].cpu().numpy())  # 转换为 numpy
                trues.append(true.reshape(1, -1))  # 保留二维形状
            
            predictions.append(pred)
            backcasts.append(backcast)
    
    # 确保 predictions 和 trues 的形状一致
    predictions = np.concatenate(predictions, axis=0)  # 拼接时保持二维形状
    backcasts = np.concatenate(backcasts, axis=0)  # 拼接时保持二维形状
    trues = np.concatenate(trues, axis=0)  # 拼接时保持二维形状
    print(f"Final predictions shape: {predictions.shape}")
    print(f"Final backcasts shape: {backcasts.shape}")
    print(f"Final trues shape: {trues.shape}")
    
    # 绘制真实值、预测值和 backcast 的曲线
    if plot_results:
        plt.figure(figsize=(12, 6))
        for i in range(min(5, len(predictions))):  # 绘制前 5 个样本
            # 构造完整的时间序列
            full_series = np.concatenate([backcasts[i], predictions[i]])
            groundtruth_series = np.concatenate([backcasts[i], trues[i]])
            
            # 绘制曲线
            plt.plot(full_series, label=f"Predicted {i+1}", linestyle='-')
            plt.plot(groundtruth_series, label=f"Ground Truth {i+1}", linestyle='--')
        plt.title("Full Series: Backcast + Forecast")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.legend()
        plt.show()
    
    return smape(trues, predictions), np.sqrt(np.mean((trues - predictions)**2))
