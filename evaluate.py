import torch
import numpy as np
from utils.metrics import compute_metrics
from model.msignn import MSI_GNN
from data.dataset import load_features_labels
from data.graph_construction import build_rbf_knn_graph
from config import Config
from utils.seed import set_seed

def evaluate_independent(config=Config, ckpt_path=None):
    set_seed(config.seed)
    # load training data to build scaler/graph if desired (we assume independent set separate)
    # load independent set
    var_ids, X_indep, y_indep, scaler = load_features_labels(config.indep_features_csv, config.indep_labels_csv)
    # normalize using independent scaler (or you can reuse training scaler if saved)
    # build graph for independent set
    edge_index, edge_weight, gamma = build_rbf_knn_graph(X_indep, k=config.knn_k, gamma=config.rbf_gamma)
    print(f"Indep graph: gamma {gamma:.3e}, edges {edge_index.size(1)}")
    # load model
    config.input_dim = X_indep.shape[1]
    device = config.device
    model = MSI_GNN(in_dim=config.input_dim, config=config).to(device)
    if ckpt_path is None:
        raise ValueError("Provide ckpt_path to evaluate")
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_indep, dtype=torch.float32, device=device)
        logits, _ = model(x_t, edge_index.to(device))
        probs = torch.sigmoid(logits.cpu().numpy())
    met = compute_metrics(y_indep, probs)
    print("Independent test metrics:", met)
    return met, probs
