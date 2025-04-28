import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # используем GCNConv для графовой агрегации
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GNNTransformerModel(nn.Module):
    def __init__(self, node_feature_dim, semantic_feature_dim, hidden_dim, n_classes, n_heads=4, n_layers=2):
        super(GNNTransformerModel, self).__init__()

        # --- GNN блок ---
        self.gnn1 = GCNConv(node_feature_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)

        # --- Transformer блок ---
        encoder_layers = TransformerEncoderLayer(d_model=semantic_feature_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=n_layers)

        # --- Финальный линейный классификатор ---
        self.fc = nn.Linear(hidden_dim + semantic_feature_dim, n_classes)

    def forward(self, data):
        # GNN часть
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))

        # Transformer часть
        semantic_input = data.semantic_features  # ожидаем размер (batch_size, seq_len, semantic_feature_dim)
        semantic_input = semantic_input.permute(1, 0, 2)  # для Transformer: (seq_len, batch_size, feature_dim)
        semantic_output = self.transformer_encoder(semantic_input)
        semantic_output = semantic_output.mean(dim=0)  # усредняем по временной оси

        # Объединение признаков
        combined = torch.cat([x, semantic_output], dim=1)

        # Финальный прогноз
        out = self.fc(combined)
        return out
