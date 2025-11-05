from pathlib import Path
import torch

ROOT = Path(__file__).parent

class Config:
    # data
    data_dir = ROOT / "data"
    features_csv = data_dir / "features.csv"    # index column = variant_id, numeric features columns after
    labels_csv = data_dir / "labels.csv"        # index column = variant_id, column 'label' (0/1)
    indep_features_csv = data_dir / "indep_features.csv"
    indep_labels_csv = data_dir / "indep_labels.csv"

    # graph
    knn_k = 8
    rbf_gamma = None   # if None auto-estimate

    # model
    input_dim = None   # infer from data
    hidden_dim = 128
    transformer_heads = 8
    transformer_layers = 2
    use_feature_mha = True   # True: feature-level MultiHeadAttention, False: feedforward-only TransformerVariant
    gnn_hidden = 128
    gat_heads = 4
    dropout = 0.3

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    lr = 5e-4
    weight_decay = 1e-2
    batch_size = 64   # full-graph training ignores this, left for compatibility
    epochs = 100
    patience = 10     # early stopping (val AUC)
    cv_folds = 5

    # misc
    save_dir = ROOT / "checkpoints"
    save_dir.mkdir(exist_ok=True)
