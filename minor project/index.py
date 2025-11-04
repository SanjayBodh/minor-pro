

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def generate_iot_graph(num_nodes=10, num_features=5, anomaly_ratio=0.2):
    """
    Creates a random IoT network graph with normal and anomalous devices.
    """

    x = torch.randn((num_nodes, num_features))


    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() > 0.7:
                edges.append([i, j])

    if not edges:

        edges.append([0, 1])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


    labels = torch.zeros(num_nodes, dtype=torch.long)
    num_anomalies = max(1, int(anomaly_ratio * num_nodes))
    anomaly_indices = np.random.choice(num_nodes, num_anomalies, replace=False)
    labels[anomaly_indices] = 1

    data = Data(x=x, edge_index=edge_index, y=labels)
    return data, edges, anomaly_indices


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x


def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            pred = out.argmax(dim=1)
            acc = (pred == data.y).sum().item() / data.num_nodes
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")
    return model


def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
        correct = (preds == data.y).sum().item()
        acc = correct / data.num_nodes
        print("\n--- Evaluation Summary ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Predicted anomalies: {preds.sum().item()}")
        print(f"Actual anomalies: {data.y.sum().item()}")
    return preds


def visualize_graph(edges, preds, anomaly_indices):
    G = nx.Graph()
    G.add_edges_from(edges)
    num_nodes = len(preds)

    color_map = []
    for i in range(num_nodes):
        if i in anomaly_indices:
            color_map.append('orange')  # True anomaly
        elif preds[i] == 1:
            color_map.append('red')     # Predicted anomaly
        else:
            color_map.append('green')   # Normal

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=700, font_color='white')
    plt.title("IoT Device Graph\n🟢 Normal | 🔴 Predicted Anomaly | 🟠 True Anomaly")
    plt.show()


def save_model(model, path="gcn_iot_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, in_channels, hidden_channels):
    model = GCN(in_channels, hidden_channels)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    print(" Generating IoT device graph...")
    data, edges, anomaly_indices = generate_iot_graph(num_nodes=12, num_features=5)

    model = GCN(in_channels=data.num_features, hidden_channels=16)

    print("\n🔧 Training model...")
    model = train_model(model, data, epochs=100, lr=0.01)

    print("\n Evaluating model...")
    preds = evaluate_model(model, data)

    print("\n Visualizing results...")
    visualize_graph(edges, preds, anomaly_indices)

    print("\n💾 Saving trained model...")
    save_model(model)
