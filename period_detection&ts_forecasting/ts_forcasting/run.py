import argparse
import pandas as pd
import torch as t
from data_process import create_dataloaders
from train import train_model
from inference import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/haochengzhang/Downloads/nbeats预测/test_data.csv")
    parser.add_argument("--input_size", type=int, default=50)
    parser.add_argument("--forecast_size", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
    # Load data using pandas
    df = pd.read_csv(args.data_path)  # 保持 df 为 Pandas DataFrame
    
    # Create dataloaders
    train_loader, val_loader, scaler = create_dataloaders(
        df, args.input_size, args.forecast_size, args.batch_size  # 直接传递 DataFrame
    )
    
    # Train
    model = train_model(train_loader, val_loader, device, 
                       args.input_size, args.forecast_size, args.epochs)
    
    # Final evaluation
    model.load_state_dict(t.load("best_model.pth"))
    smape_score, rmse = evaluate_model(model, val_loader, device, scaler, plot_results=True)
    print(f"\nFinal Evaluation | SMAPE: {smape_score:.2f}% | RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()