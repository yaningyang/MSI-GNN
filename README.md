# MSI-GNN
## 📘 Overview
**MSI-GNN** is a graph neural network framework designed for the **pathogenicity prediction of microsatellite insertions (MSIs)**.  
It integrates **multi-omic annotation features**, **similarity graph construction**, and **Transformer-enhanced attention mechanisms** to achieve accurate and interpretable predictions.

The model represents each MSI as a node in a **similarity graph**, where edge weights reflect functional similarity between variants computed via the **RBF kernel**.  

MSI-GNN combines **Graph Attention Networks (GAT)** and a **Transformer-variant encoder** to capture both **local neighborhood interactions** and **high-dimensional intra-feature dependencies**.

