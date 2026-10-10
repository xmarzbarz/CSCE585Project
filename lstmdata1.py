import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOOKBACK = 30
HORIZON = 7
RANDOM_SEED = 42

FEATURES = [
    "log_sales", "stock_hour6_22_cnt", "discount", "holiday_flag", "activity_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level", "day_sin", "day_cos",
]

REQUIRED_COLUMNS = [
    "city_id", "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt",
    "discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature",
    "avg_humidity", "avg_wind_level",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.parquet", help="Path to FreshRetailNet train.parquet")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-samples", type=int, default=250000,
                        help="Maximum training windows; use 0 for all windows")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="pytorch_lstm_output")
    parser.add_argument("--include-stockout-targets", action="store_true")
    return parser.parse_args()


def add_features(data):
    data = data.copy()
    data["dt"] = pd.to_datetime(data["dt"])
    data = data.sort_values(["city_id", "store_id", "product_id", "dt"])

    data["target_log_sales"] = np.log1p(data["sale_amount"].clip(lower=0))
    data["log_sales"] = data["target_log_sales"]
    data["target_has_stockout"] = data["stock_hour6_22_cnt"] > 0
    data["day_sin"] = np.sin(2 * np.pi * data["dt"].dt.dayofweek / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["dt"].dt.dayofweek / 7)
    data["store_code"], _ = pd.factorize(data["store_id"], sort=True)
    data["product_code"], _ = pd.factorize(data["product_id"], sort=True)

    numeric = [name for name in FEATURES if name not in {"log_sales", "day_sin", "day_cos"}]
    data[numeric] = data[numeric].fillna(0.0)
    return data


def fit_scaler(train_rows):
    scaler = {}
    for name in FEATURES:
        mean = float(train_rows[name].mean())
        std = float(train_rows[name].std())
        scaler[name] = (mean, std if std > 0 else 1.0)
    return scaler


def apply_scaler(data, scaler):
    data = data.copy()
    for name, (mean, std) in scaler.items():
        data[name] = (data[name] - mean) / std
    return data


def build_windows(data, cutoff, lookback, horizon, max_samples, include_stockout_targets, split):
    sequences, stores, products, labels = [], [], [], []
    groups = list(data.groupby(["city_id", "store_id", "product_id"], sort=False))
    random.Random(RANDOM_SEED).shuffle(groups)

    for _, group in groups:
        group = group.sort_values("dt").reset_index(drop=True)
        if len(group) < lookback + horizon:
            continue

        values = group[FEATURES].to_numpy(dtype=np.float32)
        targets = group["target_log_sales"].to_numpy(dtype=np.float32)
        dates = group["dt"].to_numpy()
        target_stockout = group["target_has_stockout"].to_numpy()
        store = int(group["store_code"].iloc[0])
        product = int(group["product_code"].iloc[0])

        for start in range(len(group) - lookback - horizon + 1):
            end = start + lookback + horizon
            future_dates = dates[start + lookback:end]
            future_stockout = target_stockout[start + lookback:end]
            if split == "train" and future_dates.max() > cutoff:
                continue
            if split == "validation" and future_dates.min() <= cutoff:
                continue
            if not include_stockout_targets and future_stockout.any():
                continue

            sequences.append(values[start:start + lookback])
            stores.append(store)
            products.append(product)
            labels.append(targets[start + lookback:end])
            if max_samples and len(labels) >= max_samples:
                break
        if max_samples and len(labels) >= max_samples:
            break

    if not labels:
        raise ValueError(f"No {split} windows were created. Inspect the data or reduce lookback/horizon.")

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(stores, dtype=np.int64),
        np.asarray(products, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


class LSTMDemandModel(nn.Module):
    def __init__(self, n_features, n_stores, n_products, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.20)
        self.store_embedding = nn.Embedding(n_stores, 8)
        self.product_embedding = nn.Embedding(n_products, 16)
        self.dense = nn.Sequential(nn.Linear(64 + 8 + 16, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, sequence, store, product):
        _, (hidden, _) = self.lstm(sequence)
        lstm_output = self.dropout(hidden[-1])
        combined = torch.cat(
            [lstm_output, self.store_embedding(store), self.product_embedding(product)], dim=1
        )
        return self.dense(combined)


def make_loader(arrays, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1]),
        torch.from_numpy(arrays[2]), torch.from_numpy(arrays[3]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss, total_rows = 0.0, 0
    predictions, actuals = [], []
    with torch.no_grad():
        for sequence, store, product, target in loader:
            sequence, store, product, target = (tensor.to(device) for tensor in (sequence, store, product, target))
            output = model(sequence, store, product)
            total_loss += loss_function(output, target).item() * len(target)
            total_rows += len(target)
            predictions.append(output.cpu().numpy())
            actuals.append(target.cpu().numpy())
    return total_loss / total_rows, np.vstack(predictions), np.vstack(actuals)


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = add_features(pd.read_parquet(args.data, columns=REQUIRED_COLUMNS))
    dates = np.sort(data["dt"].unique())
    cutoff = dates[int(len(dates) * 0.80) - 1]
    scaler = fit_scaler(data[data["dt"] <= cutoff])
    data = apply_scaler(data, scaler)

    train = build_windows(data, cutoff, args.lookback, args.horizon, args.max_samples,
                          args.include_stockout_targets, "train")
    validation_limit = max(1, args.max_samples // 4) if args.max_samples else 0
    validation = build_windows(data, cutoff, args.lookback, args.horizon, validation_limit,
                               args.include_stockout_targets, "validation")
    train_loader = make_loader(train, args.batch_size, shuffle=True)
    validation_loader = make_loader(validation, args.batch_size, shuffle=False)

    model = LSTMDemandModel(len(FEATURES), int(data["store_code"].max()) + 1,
                            int(data["product_code"].max()) + 1, args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    best_validation_loss = float("inf")
    patience, no_improvement = 3, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, rows = 0.0, 0
        for sequence, store, product, target in train_loader:
            sequence, store, product, target = (tensor.to(device) for tensor in (sequence, store, product, target))
            optimizer.zero_grad()
            output = model(sequence, store, product)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target)
            rows += len(target)

        validation_loss, _, _ = evaluate(model, validation_loader, loss_function, device)
        print(f"Epoch {epoch:02d}: train loss={running_loss / rows:.4f}, validation loss={validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss, no_improvement = validation_loss, 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lstm.pt"))
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_lstm.pt"), map_location=device))
    _, predicted_log, actual_log = evaluate(model, validation_loader, loss_function, device)
    mae = np.mean(np.abs(np.expm1(actual_log) - np.maximum(0, np.expm1(predicted_log))))
    print(f"Validation MAE across the {args.horizon}-day horizon: {mae:.2f} sale units")

    torch.save({"model_state": model.state_dict(), "features": FEATURES, "scaler": scaler,
                "lookback": args.lookback, "horizon": args.horizon},
               os.path.join(args.output_dir, "final_lstm_bundle.pt"))


if __name__ == "__main__":
    main()