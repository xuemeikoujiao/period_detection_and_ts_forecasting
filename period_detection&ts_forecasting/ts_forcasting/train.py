import torch as t
from model import create_model
from data_process import create_dataloaders
def train_model(train_loader, val_loader, device, input_size, forecast_size, epochs=100):
    model = create_model(device, input_size, forecast_size)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)  # 初始化优化器
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每 10 个 epoch 将学习率乘以 0.5
    loss_fn = t.nn.MSELoss()
    
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            backcast, forecast = model(x, t.ones_like(x))  # 获取 backcast 和 forecast
            loss_backcast = loss_fn(backcast, x)  # backcast 的损失
            loss_forecast = loss_fn(forecast, y)  # forecast 的损失
            loss = loss_backcast + loss_forecast  # 总损失
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证模型
        val_loss = evaluate(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val:
            t.save(model.state_dict(), "best_model.pth")
            best_val = val_loss
    
    return model

# def evaluate(model, val_loader, device, loss_fn):
#     model.eval()
#     total_loss = 0
#     with t.no_grad():
#         for x, y, _ in val_loader:
#             x, y = x.to(device), y.to(device)
#             pred = model(x, t.ones_like(x))
#             total_loss += loss_fn(pred, y).item()
#     return total_loss / len(val_loader)
def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with t.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            backcast, forecast = model(x, t.ones_like(x))  # 获取 backcast 和 forecast
            loss_backcast = loss_fn(backcast, x)  # backcast 的损失
            loss_forecast = loss_fn(forecast, y)  # forecast 的损失
            loss = loss_backcast + loss_forecast  # 总损失
            total_loss += loss.item()
    return total_loss / len(val_loader)