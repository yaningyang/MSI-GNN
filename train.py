import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from tqdm import tqdm
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.early_stopping import EarlyStopping
from data.dataset import load_features_labels
from data.graph_construction import build_rbf_knn_graph
from model.msignn import MSI_GNN
from config import Config

def train_cv(config=Config):
    set_seed(config.seed)
    variant_ids, X, y, scaler = load_features_labels(config.features_csv, config.labels_csv)
    config.input_dim = X.shape[1]
    # build graph once on full dataset (transductive)
    edge_index, edge_weight, gamma = build_rbf_knn_graph(X, k=config.knn_k, gamma=config.rbf_gamma)
    print(f"RBF gamma used: {gamma:.3e}, edges: {edge_index.size(1)}")
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []
    histories = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"=== Fold {fold}/{config.cv_folds} ===")
        device = config.device
        model = MSI_GNN(in_dim=config.input_dim, config=config).to(device)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        early = EarlyStopping(patience=config.patience, mode='max')
        best_state = None
        best_auc = -1
        for epoch in range(1, config.epochs+1):
            model.train()
            optimizer.zero_grad()
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            logits, _ = model(x_t, edge_index.to(device))
            loss = criterion(logits[train_idx], torch.tensor(y[train_idx], dtype=torch.float32, device=device))
            loss.backward()
            optimizer.step()
            # validate
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x_t, edge_index.to(device))
                val_probs = torch.sigmoid(val_logits[val_idx].cpu().numpy())
                metrics = compute_metrics(y[val_idx], val_probs)
                val_auc = metrics['auc']
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} loss {loss.item():.4f} val_auc {val_auc:.4f} val_f1 {metrics['f1']:.4f}")
            # early stopping
            stopped = early.step(val_auc)
            if val_auc > best_auc + 1e-5:
                best_auc = val_auc
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            if stopped:
                print("Early stopping triggered.")
                break
        # evaluate on val with best state
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            logits, _ = model(x_t, edge_index.to(device))
            probs = torch.sigmoid(logits[val_idx].cpu().numpy())
        met = compute_metrics(y[val_idx], probs)
        print("Fold result:", met)
        fold_metrics.append(met)
        histories.append({'state_dict': best_state, 'val_idx': val_idx})
        # save model checkpoint
        torch.save(best_state, os.path.join(config.save_dir, f"msignn_fold{fold}.pt"))
    # aggregate
    agg = {}
    keys = fold_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[k] = (np.mean(vals), np.std(vals))
    return agg, fold_metrics, histories, (edge_index, edge_weight)
