import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_features_labels(features_csv, labels_csv):
    """
    features_csv: path to CSV where first column is index variant_id and rest numeric features
    labels_csv: CSV with index variant_id and 'label' column
    returns: variant_ids, X (n,d), y (n,)
    """
    feats = pd.read_csv(features_csv, index_col=0)
    labs = pd.read_csv(labels_csv, index_col=0)
    # join to ensure alignment
    df = feats.join(labs, how='inner')
    if 'label' not in df.columns:
        raise ValueError("labels_csv must contain column 'label'")
    variant_ids = df.index.to_list()
    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values.astype(int)
    # z-score normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return variant_ids, X, y, scaler
