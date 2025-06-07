import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from preprocess import DataPreprocessor
from model import create_model


def build_ensemble(sequence_length: int = 24, hidden_size: int = 64, num_layers: int = 2,
                   num_epochs: int = 50) -> None:
    preprocessor = DataPreprocessor()
    features_df, targets_df = preprocessor.get_base_dataset()

    split = int(0.8 * len(features_df))

    X_train = features_df.iloc[:split]
    X_val = features_df.iloc[split:]
    y_train = targets_df.iloc[:split]
    y_val = targets_df.iloc[split:]

    # ----- train decision tree weak models -----
    tree_models = {}
    pred_full = pd.DataFrame(index=features_df.index)
    for col in targets_df.columns:
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train[col])
        tree_models[col] = dt
        pred_full[col] = dt.predict(features_df)

    # ----- conviction model -----
    future_return = np.log(targets_df["close"]) - np.log(features_df["close"])
    vol_30d = np.log(features_df["close"]).diff().rolling(24 * 30).std()
    conviction_label = (future_return >= 0.25 * vol_30d).astype(int).fillna(0)
    clf = LogisticRegression(max_iter=200)
    clf.fit(pred_full.iloc[:split], conviction_label.iloc[:split])
    conviction_prob = clf.predict_proba(pred_full)[:, 1]

    # ----- final feature set for LSTM -----
    final_df = pd.concat([features_df.reset_index(drop=True),
                          pred_full.add_prefix("pred_").reset_index(drop=True)],
                         axis=1)
    final_df["conviction"] = conviction_prob

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(final_df)
    scaled_target = target_scaler.fit_transform(targets_df[["tvl_usd"]])

    X_seq, y_seq = [], []
    for i in range(len(final_df) - sequence_length):
        X_seq.append(scaled_features[i:i+sequence_length])
        y_seq.append(scaled_target[i+sequence_length])

    X_seq = torch.FloatTensor(np.array(X_seq))
    y_seq = torch.FloatTensor(np.array(y_seq))

    train_dataset = TensorDataset(X_seq[:split-sequence_length], y_seq[:split-sequence_length])
    val_dataset = TensorDataset(X_seq[split-sequence_length:], y_seq[split-sequence_length:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = create_model(input_size=X_seq.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss += criterion(out, yb).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), "ensemble_lstm.pth")
    print("Saved ensemble model to ensemble_lstm.pth")


if __name__ == "__main__":
    build_ensemble()
